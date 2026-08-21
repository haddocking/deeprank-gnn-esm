import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

CWD = Path(__file__).resolve().parent
PDB_1ATN_1W = CWD / "data" / "pdb" / "1ATN" / "1ATN_1w.pdb"
PDB_1ATN_2W = CWD / "data" / "pdb" / "1ATN" / "1ATN_2w.pdb"

EXPECTED_ROWS = {
    ("1ATN_1w", "A", "B"): 0.021,
    ("1ATN_2w", "A", "B"): 0.008,
}


@pytest.mark.e2e
class TestMainEndToEnd(unittest.TestCase):
    """Runs the real CLI (ESM embeddings, graph generation, GNN inference)
    end to end on multiple input files and checks the predicted fnat per
    chain pair against known values."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(cls.tmpdir.name)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "deeprank_gnn.main",
                str(PDB_1ATN_1W),
                str(PDB_1ATN_2W),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=600,
        )
        cls.result = result

        csv_output = tmp_path / "GNN_esm_prediction.csv"
        cls.rows = (
            csv_output.read_text().strip().splitlines() if csv_output.exists() else []
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_exits_successfully(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)

    def test_output_has_expected_header(self):
        self.assertTrue(self.rows, "no output CSV produced")
        self.assertEqual(self.rows[0], "pdb_id,chain_i,chain_j,predicted_fnat")

    def test_every_expected_pair_is_predicted_within_tolerance(self):
        actual = {}
        for row in self.rows[1:]:
            pdb_id, chain_i, chain_j, fnat = row.split(",")
            actual[(pdb_id, chain_i, chain_j)] = float(fnat)

        self.assertEqual(set(actual), set(EXPECTED_ROWS))
        for key, expected_fnat in EXPECTED_ROWS.items():
            self.assertAlmostEqual(actual[key], expected_fnat, places=2, msg=key)


if __name__ == "__main__":
    unittest.main()
