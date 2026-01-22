# Command line interface for predicting fnat
import warnings

from Bio import BiopythonWarning

# Place warning filters BEFORE any Biopython imports that trigger warnings
warnings.filterwarnings("ignore", category=BiopythonWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

import argparse
import importlib.util
import logging
import os
import re
import shutil
import tempfile
from io import TextIOWrapper
from pathlib import Path
from typing import List

import torch
from Bio.PDB import PDBIO, Chain, Model, PDBParser, Structure
from esm import FastaBatchedDataset, pretrained

from deeprank_gnn.ginet import GINet
from deeprank_gnn.GraphGenMP import GraphHDF5
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.tools.hdf5_to_csv import hdf5_to_csv

# Configure logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    " %(asctime)s %(module)s:%(lineno)d %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)
log.addHandler(ch)


ESM_MODEL = "esm2_t33_650M_UR50D"

spec = importlib.util.find_spec("deeprank_gnn")
if spec and spec.origin:
    deeprank_gnn_path = os.path.dirname(spec.origin)
    data_path = os.path.join(deeprank_gnn_path, "data")
    GNN_ESM_MODEL = os.path.join(
        data_path, "treg_yfnat_b64_e20_lr0.001_foldall_esm.pth.tar"
    )

TOKS_PER_BATCH = 4096
REPR_LAYERS = [33]
TRUNCATION_SEQ_LENGTH = 2500
INCLUDE = ["mean", "per_tok"]
MAX_cores = 50
BATCH_SIZE = 64
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"  # configurable

###########################################################


def setup_workspace(identificator: str) -> Path:
    """Create a temporary directory for storing intermediate files."""
    cwd = Path.cwd()
    workspace = cwd / identificator

    log.info(f"Setting up workspace - {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    # else:
    #     log.info(f"WARNING: {workspace} already exists!")
    return workspace


def renumber_pdb(pdb_file_path: Path, chain_ids: list) -> None:
    """Renumber PDB file starting from 1 with no gaps."""
    log.info(f"Renumbering PDB file.")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_file_path)

    # Create a new structure with a new model
    new_structure = Structure.Structure("renumbered_structure")
    new_model = Model.Model(0)
    new_structure.add(new_model)

    # Get the chain IDs
    new_chain_ids = ["A", "B"]

    for index, chain_id in enumerate(chain_ids):
        # Get the chain
        chain = structure[0][chain_id]
        chain.parent.detach_child(chain.id)

        # Create a new chain with a new ID
        new_chain = Chain.Chain(chain_id)

        # Renumber residues in the chain starting from 1
        residue_number = 1
        for res in chain:
            res = res.copy()
            h, num, ins = res.id
            res.id = (h, residue_number, ins)
            new_chain.add(res)
            residue_number += 1

        # Add the new chain to the new model
        new_chain.id = new_chain_ids[index]
        new_model.add(new_chain)

    # Save the modified structure to the same file path
    with open(pdb_file_path, "w") as pdb_file:
        io = PDBIO()
        io.set_structure(new_structure)
        io.save(pdb_file)


def three_to_one() -> dict:
    """three_to_one mapping of 20 standard amino acids."""
    dict_ = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }

    return dict_


def pdb_to_fasta(pdb_file_path: Path, main_fasta_fh: TextIOWrapper) -> None:
    """Convert a PDB file to a FASTA file."""
    log.info(f"Reading sequence of PDB {pdb_file_path.name}")
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file_path)

    for chain_id in ["A", "B"]:
        try:
            chain = structure[0][chain_id]
        except KeyError:
            continue  # Chain might not be present

        sequence = ""
        modified_residue_count = 0  # Track modified residues

        # Get the sequence of the chain
        for residue in chain:
            resname = residue.get_resname()
            try:
                res_dict = three_to_one()
                sequence += res_dict[resname]
            except KeyError:
                sequence += "X"  # Unknown or modified residue
                modified_residue_count += 1

        # Add error/warning message if modified residues were found
        if modified_residue_count > 0:
            log.info(
                f"{modified_residue_count} unrecognized residues found in chain {chain.id} "
                f"of PDB {pdb_file_path.name}. Use DeepRank-GNN-esm with caution: non-standard residues are not officially supported."
            )

        # Write the sequence to a FASTA file
        root = re.findall(r"(.*).pdb", pdb_file_path.name)[0]
        main_fasta_fh.write(f">{root}.{chain.id}\n{sequence}\n")


# Helper function to process each batch
def process_batch(labels, strs, output_dir, representations):
    embedd_path = []
    for i, label in enumerate(labels):
        result = {"label": label}
        truncate_len = min(TRUNCATION_SEQ_LENGTH, len(strs[i]))

        # Add representations
        if "per_tok" in INCLUDE:
            result["representations"] = {
                layer: t[i, 1 : truncate_len + 1].clone()
                for layer, t in representations.items()
            }

        # Add mean representations
        if "mean" in INCLUDE:
            result["mean_representations"] = {
                layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                for layer, t in representations.items()
            }

        # Save the result to file
        output_file = output_dir / f"{label}.pt"
        torch.save(result, output_file)
        embedd_path.append(output_file)
    return embedd_path


# Helper function to get the model output
def get_model_output(toks, model, repr_layers):
    out = model(toks, repr_layers=repr_layers, return_contacts="contacts" in INCLUDE)
    representations = {layer: t.cpu() for layer, t in out["representations"].items()}
    return representations


# Main function
def get_embedding(fasta_file: Path, output_dir: Path) -> List[Path]:
    """Generate protein sequence embeddings."""
    log.info("Generating embedding for protein sequence.")
    log.info("#" * 80)

    model, alphabet = pretrained.load_model_and_alphabet(ESM_MODEL)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # Prepare dataset and data loader
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(TOKS_PER_BATCH, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(TRUNCATION_SEQ_LENGTH),
        batch_sampler=batches,
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define representation layers
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in REPR_LAYERS
    ]

    embedd_path = []
    with torch.no_grad():
        for _, (labels, strs, toks) in enumerate(data_loader):
            if torch.cuda.is_available():
                toks = toks.to("cuda", non_blocking=True)

            # Get model output and representations
            representations = get_model_output(toks, model, repr_layers)

            # Process the batch
            embedd_path.extend(
                process_batch(
                    labels,
                    strs,
                    output_dir,
                    representations,
                )
            )

    log.info("#" * 80)
    return embedd_path


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


def parse_output(csv_output: str, workspace_path: Path, chain_ids: list) -> None:
    """Parse the csv output and return the predicted fnat."""
    _data = []
    with open(csv_output, "r") as f:
        for line in f.readlines():
            if line.startswith(","):
                # this is a header
                continue
            data = line.split(",")
            pdb_id = re.findall(r"b'(.*)'", str(data[3]))[0]
            predicted_fnat = float(data[5])
            log.info(
                f"Predicted fnat for {pdb_id} between chain{chain_ids[0]} and chain{chain_ids[1]}: {predicted_fnat:.3f}"
            )
            _data.append([pdb_id, predicted_fnat])

    # output_fname = Path(workspace_path, "output.csv")
    with open(csv_output, "w") as f:
        f.write("pdb_id,predicted_fnat\n")
        for entry in _data:
            pdb, fnat = entry
            f.write(f"{pdb},{fnat:.3f}\n")

    log.info(f"Output written to {csv_output}")


def split_input_pdb(pdb_file: Path) -> list[Path]:
    """Saves PDB models in a split folder, handling both single and multiple models."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    output_dir = pdb_file.parent / f"{pdb_file.stem}_split"
    output_dir.mkdir(exist_ok=True)

    saved_files = []
    io = PDBIO()

    # Check if the PDB contains multiple models
    is_ensemble = len(list(structure)) > 1

    for model in structure:
        model_id = model.id if is_ensemble else 0  # Use 0 for single structure
        output_file = output_dir / f"{pdb_file.stem}_model_{model_id}.pdb"
        io.set_structure(model)
        io.save(str(output_file))
        saved_files.append(Path(output_file))

    log.info(f"Processed {len(saved_files)} model(s) from {pdb_file.name}")
    return saved_files


def main():
    """Main function."""

    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_file", help="Path to the PDB file.")
    parser.add_argument("chain_id_1", help="First chain ID.")
    parser.add_argument("chain_id_2", help="Second chain ID.")
    parser.add_argument("num_cores", type=int, help="Number of cores to use")
    args = parser.parse_args()

    pdb_file = args.pdb_file
    chain_id_1 = args.chain_id_1
    chain_id_2 = args.chain_id_2
    num_cores = args.num_cores

    identificator = Path(pdb_file).stem + f"-gnn_esm_pred_{chain_id_1}_{chain_id_2}"
    workspace_path = setup_workspace(identificator)

    # Copy PDB file to workspace
    src = Path(pdb_file)
    copied_pdb_file = Path(workspace_path) / Path(pdb_file).name
    shutil.copy(src, copied_pdb_file)

    pdb_files = split_input_pdb(copied_pdb_file)
    num_cores = min(num_cores, MAX_cores, len(pdb_files))
    log.info(f"Using {num_cores} cores for processing")

    embeds = []

    for pdb_file in pdb_files:
        ## Renumber PDB first!
        renumber_pdb(pdb_file_path=pdb_file, chain_ids=[chain_id_1, chain_id_2])

    ## PDB to FASTA (after renumbering)
    fasta_f = Path(workspace_path) / (pdb_files[0].stem + ".fasta")
    with open(fasta_f, "w") as f:
        pdb_to_fasta(pdb_file_path=Path(pdb_files[0]), main_fasta_fh=f)

    ## Generate embeddings
    embed_paths = get_embedding(fasta_file=fasta_f, output_dir=workspace_path)

    for pdb_file in pdb_files:
        # For pdbs that were not used in embedding cal, copy the embed and rename
        if pdb_file != pdb_files[0]:
            for embed_chain in embed_paths:
                pdb_name = pdb_file.stem
                new_stem = pdb_name + embed_chain.stem[embed_chain.stem.find(".") :]
                dst = embed_chain.with_name(new_stem + embed_chain.suffix)
                shutil.copy(embed_chain, dst)
                embeds.append(dst)

    ## Generate graph
    graph = create_graph(
        pdb_path=pdb_file.parent, workspace_path=workspace_path, nproc=num_cores
    )
    ## Predict fnat
    csv_output = predict(
        input_info=graph, workspace_path=workspace_path, ncores=num_cores
    )

    ## Present the results
    parse_output(
        csv_output=csv_output,
        workspace_path=workspace_path,
        chain_ids=[chain_id_1, chain_id_2],
    )

    # Clean copied embeddings
    for embed in embeds:
        embed.unlink()


if __name__ == "__main__":
    main()
