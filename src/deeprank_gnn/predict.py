# Command line interface for predicting fnat
import warnings

from Bio import BiopythonWarning

# Place warning filters BEFORE any Biopython imports that trigger warnings
warnings.filterwarnings("ignore", category=BiopythonWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

import importlib.util
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Optional

import h5py
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


def setup_workspace(output_dir: Optional[Path] = None) -> Path:
    """Create the workspace directory for storing intermediate files.

    If output_dir is given, it's created (if needed) and used directly, so
    intermediate files persist for inspection. Otherwise a temporary
    directory under the system temp dir is used, and left for the OS to
    clean up."""
    if output_dir is not None:
        workspace = Path(output_dir)
        workspace.mkdir(parents=True, exist_ok=True)
    else:
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


def graph_names(graph_path: str) -> set[str]:
    """Names of the graphs stored in the graph hdf5 file, i.e. of the chain
    pairs that have an interface."""
    with h5py.File(graph_path, "r") as f5:
        return set(f5.keys())


def parse_output(csv_output: str, pair_info: dict[str, tuple[str, str, str]]) -> str:
    """Rewrite csv_output in place with pdb_id/chain_i/chain_j/predicted_fnat
    columns, replacing the raw mol/prediction columns written by convert_to_csv,
    and write the per-model summary next to it: the number of scored pairs
    and the mean, standard deviation (population) and minimum of their
    predicted fnat.

    pair_info maps each graph mol name (a "{pdb_id}_{chain_i}-{chain_j}" pair
    root) to its (pdb_id, chain_i, chain_j). Every pair of pair_info is
    written, in that order; a pair without prediction (no interface between
    its chains, so no graph) is reported as NA.

    Returns the path of the summary csv.
    """
    predictions: dict[str, float] = {}
    with open(csv_output, "r") as f:
        for line in f.readlines():
            if not line.strip() or line.startswith(","):
                # empty line or header
                continue
            data = line.split(",")
            match = re.findall(r"b'(.*)'", str(data[3]))
            mol = match[0] if match else str(data[3]).strip()
            predictions[mol] = float(data[5])

    rows: list[tuple[str, str, str, Optional[float]]] = [
        (*pair_info[mol], predictions.pop(mol, None)) for mol in pair_info
    ]
    # predictions of unknown pairs (should not happen) are kept as well
    rows += [(mol, "?", "?", fnat) for mol, fnat in predictions.items()]

    model_fnats: dict[str, list[float]] = {}
    for pdb_id, chain_i, chain_j, fnat in rows:
        if fnat is None:
            log.warning(
                f"No interface found for {pdb_id} between chain {chain_i} and chain {chain_j}: fnat reported as NA"
            )
            model_fnats.setdefault(pdb_id, [])
        else:
            log.info(
                f"Predicted fnat for {pdb_id} between chain {chain_i} and chain {chain_j}: {fnat:.3f}"
            )
            model_fnats.setdefault(pdb_id, []).append(fnat)

    with open(csv_output, "w") as f:
        f.write("pdb_id,chain_i,chain_j,predicted_fnat\n")
        for pdb_id, chain_i, chain_j, fnat in rows:
            f.write(f"{pdb_id},{chain_i},{chain_j},{_format_fnat(fnat)}\n")

    log.info(f"Output written to {csv_output}")

    summary_csv = str(csv_output).replace(".csv", "_summary.csv")
    with open(summary_csv, "w") as f:
        f.write("pdb_id,scored_pairs,mean_fnat,std_fnat,min_fnat\n")
        for pdb_id, fnats in model_fnats.items():
            stats = (
                (statistics.mean(fnats), statistics.pstdev(fnats), min(fnats))
                if fnats
                else (None, None, None)
            )
            f.write(
                f"{pdb_id},{len(fnats)},{','.join(_format_fnat(x) for x in stats)}\n"
            )

    log.info(f"Summary output written to {summary_csv}")
    return summary_csv


def _format_fnat(fnat: Optional[float]) -> str:
    return "NA" if fnat is None else f"{fnat:.3f}"
