# Command line interface for predicting fnat
import warnings

from Bio import BiopythonWarning

# Place warning filters BEFORE any Biopython imports that trigger warnings
warnings.filterwarnings("ignore", category=BiopythonWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

import importlib.util
import os
import re
import tempfile
from pathlib import Path

import torch

from deeprank_gnn.ginet import GINet
from deeprank_gnn.GraphGenMP import GraphHDF5
from deeprank_gnn.logger import log
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.tools.hdf5_to_csv import hdf5_to_csv


spec = importlib.util.find_spec("deeprank_gnn")
if spec and spec.origin:
    deeprank_gnn_path = os.path.dirname(spec.origin)
    data_path = os.path.join(deeprank_gnn_path, "data")
    GNN_ESM_MODEL = os.path.join(
        data_path, "treg_yfnat_b64_e20_lr0.001_foldall_esm.pth.tar"
    )

MAX_cores = 50
BATCH_SIZE = 64

###########################################################


def setup_workspace() -> Path:
    """Create a temporary directory (under the system temp dir) for storing intermediate files."""
    workspace = Path(tempfile.mkdtemp())
    log.info(f"Setting up workspace - {workspace}")
    return workspace


def create_graph(pdb_path: Path, workspace_path: Path, nproc: int) -> str:
    """Generate a graph"""
    log.info(f"Generating graph, using {nproc} processors")

    outfile = str(workspace_path / "graph.hdf5")

    with tempfile.TemporaryDirectory() as tmpdir:
        GraphHDF5(
            pdb_path=pdb_path,
            embedding_path=workspace_path,
            graph_type="residue",
            outfile=outfile,
            nproc=nproc,
            tmpdir=tmpdir,
        )

    assert os.path.exists(outfile), f"Graph file {outfile} not found."
    log.info(f"Graph file generated: {outfile}")
    return outfile


def predict(input_info: str, workspace_path: Path, ncores: int) -> str:
    """Predict the fnat of a protein complex."""
    log.info("Predicting fnat of protein complex.")
    gnn = GINet
    target = "fnat"
    edge_attr = ["dist"]
    #
    threshold = 0.3

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device_name}")

    node_feature = ["type", "polarity", "bsa", "charge", "embedding"]
    output = str(workspace_path / "GNN_esm_prediction.hdf5")
    # with nostdout():
    model = NeuralNet(
        input_info,
        gnn,
        device_name=device_name,
        edge_feature=edge_attr,
        node_feature=node_feature,
        num_workers=ncores,
        batch_size=BATCH_SIZE,
        target=target,
        pretrained_model=GNN_ESM_MODEL,
        threshold=threshold,
    )
    model.test(hdf5=output)

    output_csv = convert_to_csv(output)

    return output_csv


def convert_to_csv(hdf5_path: str) -> str:
    """Convert the hdf5 file to csv."""
    hdf5_to_csv(hdf5_path)
    csv_path = str(hdf5_path).replace(".hdf5", ".csv")

    assert os.path.exists(csv_path), f"CSV file {csv_path} not found."

    return csv_path


def parse_output(
    csv_output: str, workspace_path: Path, pair_info: dict[str, tuple[str, str, str]]
) -> None:
    """Parse the csv output and return the predicted fnat.

    pair_info maps each graph mol name (a "{pdb_id}_{chain_i}-{chain_j}" pair
    root) to its (pdb_id, chain_i, chain_j).
    """
    _data = []
    with open(csv_output, "r") as f:
        for line in f.readlines():
            if line.startswith(","):
                # this is a header
                continue
            data = line.split(",")
            mol = re.findall(r"b'(.*)'", str(data[3]))[0]
            predicted_fnat = float(data[5])
            pdb_id, chain_i, chain_j = pair_info.get(mol, (mol, "?", "?"))
            log.info(
                f"Predicted fnat for {pdb_id} between chain{chain_i} and chain{chain_j}: {predicted_fnat:.3f}"
            )
            _data.append([pdb_id, chain_i, chain_j, predicted_fnat])

    # output_fname = Path(workspace_path, "output.csv")
    with open(csv_output, "w") as f:
        f.write("pdb_id,chain_i,chain_j,predicted_fnat\n")
        for entry in _data:
            pdb_id, chain_i, chain_j, fnat = entry
            f.write(f"{pdb_id},{chain_i},{chain_j},{fnat:.3f}\n")

    log.info(f"Output written to {csv_output}")
