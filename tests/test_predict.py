import tempfile
import unittest
from pathlib import Path

from deeprank_gnn.predict import parse_output


class TestParseOutput(unittest.TestCase):
    def test_rewrites_csv_with_chain_columns(self):
        raw_csv = (
            ",epoch,set,model,targets,prediction\n"
            "0,0,test,b'1qu9_ABC_A-B',n,0.488\n"
            "1,0,test,b'1qu9_ABC_A-C',n,0.484\n"
            ",epoch,set,model,targets,prediction\n"
            "0,0,test,b'2oob_A-B',n,0.792\n"
        )
        pair_info = {
            "1qu9_ABC_A-B": ("1qu9_ABC", "A", "B"),
            "1qu9_ABC_A-C": ("1qu9_ABC", "A", "C"),
            "2oob_A-B": ("2oob", "A", "B"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_output = Path(tmpdir) / "result.csv"
            csv_output.write_text(raw_csv)

            parse_output(
                csv_output=str(csv_output),
                workspace_path=Path(tmpdir),
                pair_info=pair_info,
            )

            lines = csv_output.read_text().splitlines()

        self.assertEqual(lines[0], "pdb_id,chain_i,chain_j,predicted_fnat")
        self.assertEqual(
            lines[1:],
            [
                "1qu9_ABC,A,B,0.488",
                "1qu9_ABC,A,C,0.484",
                "2oob,A,B,0.792",
            ],
        )

    def test_unknown_mol_falls_back_to_placeholder_chains(self):
        raw_csv = (
            ",epoch,set,model,targets,prediction\n0,0,test,b'unknown_mol',n,0.100\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_output = Path(tmpdir) / "result.csv"
            csv_output.write_text(raw_csv)

            parse_output(
                csv_output=str(csv_output), workspace_path=Path(tmpdir), pair_info={}
            )

            lines = csv_output.read_text().splitlines()

        self.assertEqual(lines[1], "unknown_mol,?,?,0.100")


if __name__ == "__main__":
    unittest.main()
