"""Sequence and MultiFasta: labeled amino-acid sequences and the deduped
multi-FASTA/ESM-embedding pipeline built from them."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from esm import FastaBatchedDataset, pretrained

log = logging.getLogger(__name__)

ESM_MODEL = "esm2_t33_650M_UR50D"
TOKS_PER_BATCH = 4096
REPR_LAYERS = [33]
TRUNCATION_SEQ_LENGTH = 2500


@dataclass
class Sequence:
    label: str
    sequence: str

    @property
    def modified_residue_count(self) -> int:
        return self.sequence.count("X")


@dataclass
class MultiFasta:
    """A deduped collection of Sequences: identical sequences (e.g. homodimer
    chains, repeats across ensemble models) are kept once, with every label
    mapped to the canonical label actually written/embedded."""

    label_map: dict[str, str] = field(
        default_factory=dict
    )  # every label -> canonical label
    sequences: dict[str, Sequence] = field(
        default_factory=dict
    )  # canonical label -> Sequence
    _seq_to_canonical_label: dict[str, str] = field(default_factory=dict, repr=False)

    def add(self, seq: Sequence) -> None:
        if seq.sequence in self._seq_to_canonical_label:
            self.label_map[seq.label] = self._seq_to_canonical_label[seq.sequence]
        else:
            self._seq_to_canonical_label[seq.sequence] = seq.label
            self.label_map[seq.label] = seq.label
            self.sequences[seq.label] = seq

    def gen_embeddings(self) -> list[tuple[str, torch.Tensor]]:
        """Generate ESM embeddings for every unique sequence in this MultiFasta.

        Runs ESM once over the in-memory sequences and returns one
        (sequence, embedding) pair per unique sequence - one ESM call per
        unique sequence, not per label.
        """
        log.info(f"Generating embeddings for {len(self.sequences)} unique sequence(s).")

        model, alphabet = pretrained.load_model_and_alphabet(ESM_MODEL)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        labels = list(self.sequences.keys())
        seqs = [seq.sequence for seq in self.sequences.values()]
        dataset = FastaBatchedDataset(labels, seqs)
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
                    results.append((self.sequences[label].sequence, embedding))

        return results

    def write_embeddings(
        self, embeddings: list[tuple[str, torch.Tensor]], output_dir: Path
    ) -> list[Path]:
        """Persist gen_embeddings() output as one {label}.pt file per label -
        including labels whose sequence was deduplicated away - matching the
        naming GraphGenMP._add_embedding expects on disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        seq_to_embedding = {sequence: embedding for sequence, embedding in embeddings}

        saved_files = []
        for label, canonical_label in self.label_map.items():
            sequence = self.sequences[canonical_label].sequence
            embedding = seq_to_embedding[sequence]

            output_file = output_dir / f"{label}.pt"
            torch.save(
                {"label": label, "representations": {REPR_LAYERS[-1]: embedding}},
                output_file,
            )
            saved_files.append(output_file)

        log.info(f"Wrote {len(saved_files)} embedding file(s) to {output_dir}")
        return saved_files
