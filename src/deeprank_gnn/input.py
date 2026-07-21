"""PDBInput: wraps a (possibly multi-model) Biopython Structure as the main
data structure carried through the DeepRank-GNN-esm prediction pipeline."""

import logging
from io import TextIOWrapper
from pathlib import Path

import torch
from Bio.PDB import PDBIO, Chain, Model, PDBParser, Structure
from esm import FastaBatchedDataset, pretrained

log = logging.getLogger(__name__)

ESM_MODEL = "esm2_t33_650M_UR50D"
TOKS_PER_BATCH = 4096
REPR_LAYERS = [33]
TRUNCATION_SEQ_LENGTH = 2500


def three_to_one() -> dict:
    """three_to_one mapping of 20 standard amino acids."""
    return {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }


class PDBInput:
    """A possibly multi-model PDB structure carried through renumbering,
    FASTA/embedding generation, and per-model materialization for graph
    generation."""

    def __init__(self, structure: Structure.Structure, name: str):
        self.structure = structure
        self.name = name
        self.label_map: dict[str, str] = {}  # set by to_fasta()
        self.sequences: dict[
            str, str
        ] = {}  # set by to_fasta(): embedded label -> sequence

    @classmethod
    def from_file(cls, pdb_file_path: Path) -> "PDBInput":
        """Parse a PDB file into a PDBInput."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_file_path.stem, pdb_file_path)
        return cls(structure=structure, name=pdb_file_path.stem)

    @property
    def models(self) -> list:
        return list(self.structure)

    @property
    def is_ensemble(self) -> bool:
        return len(self.models) > 1

    def model_name(self, model) -> str:
        """Root name used for a given model's FASTA/embedding/PDB entries."""
        return f"{self.name}_model{model.id}" if self.is_ensemble else self.name

    def renumber(self) -> "PDBInput":
        """Renumber residues in each chain of each model starting from 1
        with no gaps, mutating this instance's structure in place."""
        log.info(f"Renumbering structure {self.name}.")

        new_structure = Structure.Structure("renumbered_structure")

        for model in self.structure:
            new_model = Model.Model(model.id)

            for chain in model:
                new_chain = Chain.Chain(chain.id)

                residue_number = 1
                for res in chain:
                    res = res.copy()
                    h, num, ins = res.id
                    res.id = (h, residue_number, ins)
                    new_chain.add(res)
                    residue_number += 1

                new_model.add(new_chain)

            new_structure.add(new_model)

        self.structure = new_structure
        return self

    def to_fasta(self, main_fasta_fh: TextIOWrapper) -> dict[str, str]:
        """Write one FASTA entry per unique sequence found across all
        models/chains of this structure.

        Identical sequences (e.g. repeated across ensemble models, or
        homodimer chains) are written once. Sets and returns self.label_map,
        mapping every "{root}.{chain_id}" label to the label actually
        written to the FASTA (and thus embedded) for its sequence.
        """
        log.info(f"Reading sequence of structure {self.name}")

        seq_to_label: dict[str, str] = {}  # sequence -> label written to the FASTA
        label_map: dict[str, str] = {}  # every label -> label actually embedded

        for model in self.models:
            root = self.model_name(model)

            for chain in model:
                sequence = ""
                modified_residue_count = 0  # Track modified residues

                for residue in chain:
                    resname = residue.get_resname()
                    try:
                        sequence += three_to_one()[resname]
                    except KeyError:
                        sequence += "X"  # Unknown or modified residue
                        modified_residue_count += 1

                if modified_residue_count > 0:
                    log.info(
                        f"{modified_residue_count} unrecognized residues found in chain {chain.id} "
                        f"of {root}. Use DeepRank-GNN-esm with caution: non-standard residues are not officially supported."
                    )

                label = f"{root}.{chain.id}"

                if sequence in seq_to_label:
                    # Same sequence already written under another label - reuse its embedding.
                    label_map[label] = seq_to_label[sequence]
                else:
                    seq_to_label[sequence] = label
                    label_map[label] = label
                    self.sequences[label] = sequence
                    main_fasta_fh.write(f">{label}\n{sequence}\n")

        self.label_map = label_map
        return label_map

    def gen_embeddings(self, workspace_path: Path) -> list[tuple[str, torch.Tensor]]:
        """Generate ESM embeddings for every unique sequence in this structure.

        Writes the deduplicated multi-FASTA to workspace_path (via to_fasta),
        runs ESM once over it, and returns one (sequence, embedding) pair per
        unique sequence - one ESM call per unique sequence, not per chain/model.
        """
        workspace_path.mkdir(parents=True, exist_ok=True)
        fasta_path = workspace_path / f"{self.name}.fasta"

        self.sequences = {}
        with open(fasta_path, "w") as fh:
            self.to_fasta(fh)

        log.info(f"Generating embeddings for {len(self.sequences)} unique sequence(s).")

        model, alphabet = pretrained.load_model_and_alphabet(ESM_MODEL)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        dataset = FastaBatchedDataset.from_file(fasta_path)
        batches = dataset.get_batch_indices(TOKS_PER_BATCH, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(TRUNCATION_SEQ_LENGTH),
            batch_sampler=batches,
        )

        repr_layers = [
            (i + model.num_layers + 1) % (model.num_layers + 1) for i in REPR_LAYERS
        ]
        last_layer = repr_layers[-1]

        results: list[tuple[str, torch.Tensor]] = []
        with torch.no_grad():
            for labels, strs, toks in data_loader:
                if torch.cuda.is_available():
                    toks = toks.to("cuda", non_blocking=True)

                out = model(toks, repr_layers=repr_layers, return_contacts=False)
                representations = {
                    layer: t.cpu() for layer, t in out["representations"].items()
                }

                for i, label in enumerate(labels):
                    truncate_len = min(TRUNCATION_SEQ_LENGTH, len(strs[i]))
                    embedding = representations[last_layer][
                        i, 1 : truncate_len + 1
                    ].clone()
                    results.append((self.sequences[label], embedding))

        return results

    def write_embeddings(
        self, embeddings: list[tuple[str, torch.Tensor]], output_dir: Path
    ) -> list[Path]:
        """Persist gen_embeddings() output as one {root}.{chain_id}.pt file per
        model/chain label - including labels whose sequence was deduplicated
        away by to_fasta() - matching the naming GraphGenMP._add_embedding
        expects on disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        seq_to_embedding = {sequence: embedding for sequence, embedding in embeddings}

        saved_files = []
        for label, canonical_label in self.label_map.items():
            sequence = self.sequences[canonical_label]
            embedding = seq_to_embedding[sequence]

            output_file = output_dir / f"{label}.pt"
            torch.save(
                {"label": label, "representations": {REPR_LAYERS[-1]: embedding}},
                output_file,
            )
            saved_files.append(output_file)

        log.info(f"Wrote {len(saved_files)} embedding file(s) to {output_dir}")
        return saved_files

    def write_models(self, output_dir: Path) -> list:
        """Write one single-model PDB file per model to output_dir, named to
        match to_fasta()'s label roots. Needed because downstream graph
        generation (GraphHDF5/pdb2sql) reads real single-model PDB files
        from disk, not in-memory Structures."""
        output_dir.mkdir(parents=True, exist_ok=True)

        io = PDBIO()
        saved_files = []
        for model in self.models:
            root = self.model_name(model)
            output_file = output_dir / f"{root}.pdb"
            io.set_structure(model)
            io.save(str(output_file))
            saved_files.append(output_file)

        log.info(
            f"Wrote {len(saved_files)} model(s) of structure {self.name} to {output_dir}"
        )
        return saved_files
