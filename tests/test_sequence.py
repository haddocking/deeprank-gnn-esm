import tempfile
import unittest
from pathlib import Path

import torch

from deeprank_gnn.sequence import MultiFasta, Sequence


class TestSequence(unittest.TestCase):
    def test_modified_residue_count(self):
        seq = Sequence(label="a", sequence="MKXTAXYI")
        self.assertEqual(seq.modified_residue_count, 2)

    def test_modified_residue_count_zero(self):
        seq = Sequence(label="a", sequence="MKTAYI")
        self.assertEqual(seq.modified_residue_count, 0)


class TestMultiFastaAdd(unittest.TestCase):
    def test_identical_sequences_are_deduped(self):
        multi_fasta = MultiFasta()
        multi_fasta.add(Sequence(label="root.A", sequence="MKTAYI"))
        multi_fasta.add(Sequence(label="root.B", sequence="MKTAYI"))

        self.assertEqual(len(multi_fasta.sequences), 1)
        self.assertEqual(
            multi_fasta.label_map["root.A"], multi_fasta.label_map["root.B"]
        )
        self.assertEqual(multi_fasta.label_map["root.A"], "root.A")

    def test_distinct_sequences_are_kept(self):
        multi_fasta = MultiFasta()
        multi_fasta.add(Sequence(label="root.A", sequence="MKTAYI"))
        multi_fasta.add(Sequence(label="root.B", sequence="GGCLVK"))

        self.assertEqual(len(multi_fasta.sequences), 2)
        self.assertEqual(multi_fasta.label_map["root.A"], "root.A")
        self.assertEqual(multi_fasta.label_map["root.B"], "root.B")


class TestMultiFastaWriteEmbeddings(unittest.TestCase):
    def test_writes_one_file_per_label_including_deduped(self):
        multi_fasta = MultiFasta()
        multi_fasta.add(Sequence(label="root.A", sequence="MKTAYI"))
        multi_fasta.add(Sequence(label="root.B", sequence="MKTAYI"))  # dupe of A
        multi_fasta.add(Sequence(label="root.C", sequence="GGCLVK"))

        tensor_ab = torch.rand(6, 4)
        tensor_c = torch.rand(6, 4)
        embeddings = [("MKTAYI", tensor_ab), ("GGCLVK", tensor_c)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            saved_files = multi_fasta.write_embeddings(embeddings, output_dir)

            self.assertEqual(
                {p.name for p in saved_files},
                {"root.A.pt", "root.B.pt", "root.C.pt"},
            )

            for label in ("root.A", "root.B"):
                result = torch.load(output_dir / f"{label}.pt")
                self.assertTrue(torch.equal(result["representations"][33], tensor_ab))

            result_c = torch.load(output_dir / "root.C.pt")
            self.assertTrue(torch.equal(result_c["representations"][33], tensor_c))


if __name__ == "__main__":
    unittest.main()
