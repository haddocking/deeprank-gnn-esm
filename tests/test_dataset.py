import unittest
from deeprank_gnn.DataSet import HDF5DataSet, DivideDataSet, PreCluster
import os
import shutil
import tempfile

import h5py

CWD = os.path.dirname(os.path.realpath(__file__))

class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.database = f'{CWD}/data/hdf5/1ATN_residue.hdf5'

    def test_dataset(self):
        _ = HDF5DataSet(
            database=self.database,
            node_feature=["type", "polarity", "bsa", "embedding"], # type: ignore
            edge_feature=["dist"],
            target="fnat",
            index=None,
        )

    def test_dataset_filter(self):
        _ = HDF5DataSet(
            database=self.database,
            node_feature=["type", "polarity", "bsa", "embedding"], # type: ignore
            edge_feature=["dist"],
            target="fnat",
            index=None,
            dict_filter={"fnat": "<10"},
        )

    def test_precluster_drops_invalid_graphs_from_index(self):
        # a graph group without node data (e.g. a chain pair without
        # interface) must not stay in the index, or the DataLoader gets None
        with tempfile.TemporaryDirectory() as tmpdir:
            database = os.path.join(tmpdir, "graphs.hdf5")
            shutil.copy(self.database, database)
            with h5py.File(database, "a") as f5:
                f5.create_group("broken")

            dataset = HDF5DataSet(
                database=database,
                node_feature=["type", "polarity", "bsa", "embedding"],  # type: ignore
                edge_feature=["dist"],
                target="fnat",
                index=None,
            )
            self.assertIn((database, "broken"), dataset.index_complexes)

            PreCluster(dataset, method="mcl")

            self.assertNotIn((database, "broken"), dataset.index_complexes)
            self.assertTrue(dataset.index_complexes)
            self.assertTrue(all(dataset.get(i) is not None for i in range(dataset.len())))
            with h5py.File(database, "r") as f5:
                self.assertNotIn("broken", f5)


if __name__ == "__main__":
    unittest.main()
