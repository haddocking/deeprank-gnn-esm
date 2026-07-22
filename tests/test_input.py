import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from Bio.Data.PDBData import protein_letters_1to3
from Bio.PDB import PDBIO, Atom, Chain, Model, Residue, Structure

from deeprank_gnn.input import PDBInput


def _make_chain(chain_id: str, seq: str) -> Chain.Chain:
    chain = Chain.Chain(chain_id)
    for i, aa in enumerate(seq, start=1):
        resname = protein_letters_1to3[aa].upper()
        residue = Residue.Residue((" ", i, " "), resname, "")
        atom = Atom.Atom(
            "CA", np.array([float(i), 0.0, 0.0]), 20.0, 1.0, " ", "CA", i, "C"
        )
        residue.add(atom)
        chain.add(residue)
    return chain


def _make_structure(models: list) -> Structure.Structure:
    """models: list of {chain_id: sequence} dicts, one per model."""
    structure = Structure.Structure("synthetic")
    for model_id, chains in enumerate(models):
        model = Model.Model(model_id)
        for chain_id, seq in chains.items():
            model.add(_make_chain(chain_id, seq))
        structure.add(model)
    return structure


class TestPDBInputSingleModel(unittest.TestCase):
    def setUp(self):
        self.structure = _make_structure([{"A": "MKTAYI", "B": "GGCLVK"}])
        self.pdb_input = PDBInput(self.structure, name="synth")

    def test_is_ensemble_false_for_single_model(self):
        self.assertFalse(self.pdb_input.is_ensemble)

    def test_model_name_has_no_model_suffix(self):
        model = self.pdb_input.models[0]
        self.assertEqual(self.pdb_input.model_name(model), "synth")

    def test_chain_pairs_two_chains(self):
        model = self.pdb_input.models[0]
        self.assertEqual(self.pdb_input.chain_pairs(model), [("A", "B")])

    def test_pair_name(self):
        model = self.pdb_input.models[0]
        self.assertEqual(self.pdb_input.pair_name(model, "A", "B"), "synth_A-B")


class TestPDBInputEnsemble(unittest.TestCase):
    def setUp(self):
        self.structure = _make_structure(
            [{"A": "MKTAYI", "B": "GGCLVK"}, {"A": "MKTAYI", "B": "GGCLVK"}]
        )
        self.pdb_input = PDBInput(self.structure, name="synth")

    def test_is_ensemble_true_for_multiple_models(self):
        self.assertTrue(self.pdb_input.is_ensemble)

    def test_model_name_has_model_suffix(self):
        model0, model1 = self.pdb_input.models
        self.assertEqual(self.pdb_input.model_name(model0), "synth_model0")
        self.assertEqual(self.pdb_input.model_name(model1), "synth_model1")


class TestRenumber(unittest.TestCase):
    def test_renumber_closes_gaps(self):
        structure = _make_structure([{"A": "MKTAYI"}])
        # Introduce gaps/non-sequential numbering in the original residue ids
        chain = list(structure[0])[0]
        for offset, residue in enumerate(list(chain)):
            h, _, ins = residue.id
            residue.id = (h, 100 + offset * 5, ins)

        pdb_input = PDBInput(structure, name="synth").renumber()

        chain = list(pdb_input.models[0])[0]
        residue_numbers = [res.id[1] for res in chain]
        self.assertEqual(residue_numbers, list(range(1, len(residue_numbers) + 1)))


class TestToMultiFasta(unittest.TestCase):
    def test_identical_sequences_are_deduped(self):
        # Homodimer: chains A and B share the same sequence
        structure = _make_structure([{"A": "MKTAYI", "B": "MKTAYI"}])
        pdb_input = PDBInput(structure, name="synth")

        multi_fasta = pdb_input.to_multi_fasta()

        self.assertEqual(len(multi_fasta.sequences), 1)
        self.assertEqual(
            multi_fasta.label_map["synth.A"], multi_fasta.label_map["synth.B"]
        )

    def test_distinct_sequences_are_kept_separate(self):
        structure = _make_structure([{"A": "MKTAYI", "B": "GGCLVK"}])
        pdb_input = PDBInput(structure, name="synth")

        multi_fasta = pdb_input.to_multi_fasta()

        self.assertEqual(len(multi_fasta.sequences), 2)
        self.assertNotEqual(
            multi_fasta.label_map["synth.A"], multi_fasta.label_map["synth.B"]
        )


class TestWriteChainPairs(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.tmp_path = Path(self.tmpdir.name)

        self.structure = _make_structure(
            [{"A": "MKTAYI", "B": "GGCLVK", "C": "PPQRST"}]
        )
        self.pdb_input = PDBInput(self.structure, name="synth")

        # Fake per-chain embeddings, standing in for write_embeddings() output
        self.embedding_dir = self.tmp_path / "embeddings"
        self.embedding_dir.mkdir()
        self.chain_tensors = {}
        for chain_id in ("A", "B", "C"):
            tensor = torch.rand(6, 4)
            self.chain_tensors[chain_id] = tensor
            torch.save(
                {"label": f"synth.{chain_id}", "representations": {33: tensor}},
                self.embedding_dir / f"synth.{chain_id}.pt",
            )

        self.pdb_dir = self.tmp_path / "structures"
        self.pair_info = self.pdb_input.write_chain_pairs(
            self.pdb_dir, self.embedding_dir
        )

    def test_one_pdb_file_per_chain_pair(self):
        expected_roots = {"synth_A-B", "synth_A-C", "synth_B-C"}
        written = {p.stem for p in self.pdb_dir.glob("*.pdb")}
        self.assertEqual(written, expected_roots)

    def test_pair_info_keeps_original_chain_ids(self):
        self.assertEqual(self.pair_info["synth_A-B"], ("synth", "A", "B"))
        self.assertEqual(self.pair_info["synth_A-C"], ("synth", "A", "C"))
        self.assertEqual(self.pair_info["synth_B-C"], ("synth", "B", "C"))

    def test_written_pdb_chains_are_relabeled_to_A_and_B(self):
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        for root in ("synth_A-B", "synth_A-C", "synth_B-C"):
            structure = parser.get_structure(root, self.pdb_dir / f"{root}.pdb")
            chain_ids = {chain.id for chain in structure[0]}
            self.assertEqual(chain_ids, {"A", "B"})

    def test_embedding_files_copied_under_relabeled_names(self):
        # synth_A-C pairs original chain A (-> A) with original chain C (-> B)
        pt_a = torch.load(self.embedding_dir / "synth_A-C.A.pt")
        pt_b = torch.load(self.embedding_dir / "synth_A-C.B.pt")
        self.assertTrue(
            torch.equal(pt_a["representations"][33], self.chain_tensors["A"])
        )
        self.assertTrue(
            torch.equal(pt_b["representations"][33], self.chain_tensors["C"])
        )


class TestPDBInputFromFile(unittest.TestCase):
    def test_from_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            structure = _make_structure([{"A": "MKTAYI", "B": "GGCLVK"}])

            pdb_file = tmp_path / "synth.pdb"
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pdb_file))

            pdb_input = PDBInput.from_file(pdb_file)

            self.assertEqual(pdb_input.name, "synth")
            self.assertEqual(len(pdb_input.models), 1)
            chain_ids = {chain.id for chain in pdb_input.models[0]}
            self.assertEqual(chain_ids, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
