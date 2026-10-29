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

            parse_output(csv_output=str(csv_output), pair_info={})

            lines = csv_output.read_text().splitlines()

        self.assertEqual(lines[1], "unknown_mol,?,?,0.100")

    def test_pair_without_prediction_is_reported_as_na(self):
        # 1qu9_ABC_A-C has no interface, so no graph and no prediction
        raw_csv = (
            ",epoch,set,model,targets,prediction\n"
            "0,0,test,b'1qu9_ABC_A-B',n,0.488\n"
            "1,0,test,b'1qu9_ABC_B-C',n,0.100\n"
        )
        pair_info = {
            "1qu9_ABC_A-B": ("1qu9_ABC", "A", "B"),
            "1qu9_ABC_A-C": ("1qu9_ABC", "A", "C"),
            "1qu9_ABC_B-C": ("1qu9_ABC", "B", "C"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_output = Path(tmpdir) / "result.csv"
            csv_output.write_text(raw_csv)

            summary_csv = parse_output(csv_output=str(csv_output), pair_info=pair_info)

            lines = csv_output.read_text().splitlines()
            summary = Path(summary_csv).read_text().splitlines()

        self.assertEqual(
            lines[1:],
            [
                "1qu9_ABC,A,B,0.488",
                "1qu9_ABC,A,C,NA",
                "1qu9_ABC,B,C,0.100",
            ],
        )
        self.assertEqual(Path(summary_csv).name, "result_summary.csv")
        self.assertEqual(summary, ["pdb_id,combined_fnat", "1qu9_ABC,0.588"])

    def test_no_prediction_at_all(self):
        pair_info = {"apart_A-B": ("apart", "A", "B")}

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_output = Path(tmpdir) / "result.csv"
            csv_output.write_text("")

            summary_csv = parse_output(csv_output=str(csv_output), pair_info=pair_info)

            lines = csv_output.read_text().splitlines()
            summary = Path(summary_csv).read_text().splitlines()

        self.assertEqual(lines, ["pdb_id,chain_i,chain_j,predicted_fnat", "apart,A,B,NA"])
        self.assertEqual(summary, ["pdb_id,combined_fnat", "apart,NA"])


if __name__ == "__main__":
    unittest.main()
