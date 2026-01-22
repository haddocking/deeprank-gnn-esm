import tempfile
import unittest
from pathlib import Path

import torch

from deeprank_gnn.predict import get_embedding


class TestPredict(unittest.TestCase):
    def test_get_embedding(self):
        """Test that get_embedding generates embeddings correctly."""
        # Create a temporary directory for test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a simple fasta file with a short sequence
            fasta_file = tmpdir / "test.fasta"
            fasta_file.write_text(">test_seq.A\nMKTAYIAKQRQISFVKSHFSRQLE\n")

            # Create output directory
            output_dir = tmpdir / "embeddings"

            # Generate embeddings
            embed_paths = get_embedding(fasta_file, output_dir)

            # Verify output
            self.assertEqual(len(embed_paths), 1)
            self.assertTrue(embed_paths[0].exists())
            self.assertTrue(str(embed_paths[0]).endswith(".pt"))

            # Load and verify the embedding file
            result = torch.load(embed_paths[0])
            self.assertIn("label", result)
            self.assertIn("representations", result)
            self.assertIn("mean_representations", result)
            self.assertEqual(result["label"], "test_seq.A")


if __name__ == "__main__":
    unittest.main()
