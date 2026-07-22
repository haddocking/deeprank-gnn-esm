"""PDBInput: wraps a (possibly multi-model) Biopython Structure as the main
data structure carried through the DeepRank-GNN-esm prediction pipeline."""

import logging
from pathlib import Path

import torch
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import PDBIO, Model, PDBParser, Structure
from Bio.PDB.Chain import Chain

from deeprank_gnn.sequence import MultiFasta, Sequence

log = logging.getLogger(__name__)


class PDBInput:
    """A possibly multi-model PDB structure carried through renumbering,
    FASTA/embedding generation, and per-model materialization for graph
    generation."""

    def __init__(self, structure: Structure.Structure, name: str):
        self.structure = structure
        self.name = name
        self._multi_fasta = MultiFasta()  # set by to_multi_fasta()

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

    @staticmethod
    def _chain_to_seq(chain: Chain) -> str:
        sequence = ""

        for residue in chain:
            resname = residue.get_resname()
            if resname in protein_letters_3to1:
                sequence += protein_letters_3to1[resname]
            else:
                sequence += "X"  # Unknown/modified residue

        return sequence

    def to_multi_fasta(self) -> MultiFasta:
        """Build one Sequence per model/chain of this structure and dedup
        them by content into self._multi_fasta.

        Identical sequences (e.g. repeated across ensemble models, or
        homodimer chains) are kept once. self._multi_fasta.label_map maps
        every "{root}.{chain_id}" label to the label actually embedded for
        its sequence.
        """
        log.info(f"Reading sequence of structure {self.name}")

        self._multi_fasta = MultiFasta()

        for model in self.models:
            root = self.model_name(model)

            for chain in model:
                label = f"{root}.{chain.id}"
                seq_obj = Sequence(label=label, sequence=self._chain_to_seq(chain))

                if seq_obj.modified_residue_count > 0:
                    log.warning(
                        f"Unrecognized residues found in chain {chain.id} "
                        f"of {root}. Use DeepRank-GNN-esm with caution: non-standard residues are not officially supported."
                    )

                self._multi_fasta.add(seq_obj)

        return self._multi_fasta

    def gen_embeddings(self) -> list[tuple[str, torch.Tensor]]:
        """Generate ESM embeddings for every unique sequence in this structure.

        Builds the deduplicated MultiFasta (via to_multi_fasta) and delegates
        the ESM run to it - one ESM call per unique sequence, not per
        chain/model.
        """
        self.to_multi_fasta()
        return self._multi_fasta.gen_embeddings()

    def write_embeddings(
        self, embeddings: list[tuple[str, torch.Tensor]], output_dir: Path
    ) -> list[Path]:
        """Persist gen_embeddings() output as one {root}.{chain_id}.pt file per
        model/chain label - including labels whose sequence was deduplicated
        away - matching the naming GraphGenMP._add_embedding expects on disk."""
        return self._multi_fasta.write_embeddings(embeddings, output_dir)

    def write_models(self, output_dir: Path) -> list:
        """Write one single-model PDB file per model to output_dir, named to
        match to_multi_fasta()'s label roots. Needed because downstream graph
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
