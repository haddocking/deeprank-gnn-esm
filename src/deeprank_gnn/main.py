import argparse
import shutil
from pathlib import Path

from deeprank_gnn.input import PDBInput
from deeprank_gnn.logger import log
from deeprank_gnn.predict import (
    MAX_cores,
    create_graph,
    parse_output,
    predict,
    setup_workspace,
)


def main():
    """Main function."""

    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_files", nargs="+", help="Path(s) to the PDB file(s).")
    parser.add_argument(
        "--num_cores", type=int, default=1, help="Number of cores to use (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save intermediate files in (default: a temporary "
        "directory that is discarded after the run).",
    )
    args = parser.parse_args()

    pdb_files = [Path(p) for p in args.pdb_files]
    stems = [pdb_file.stem for pdb_file in pdb_files]
    if len(set(stems)) != len(stems):
        parser.error("Input PDB files must have unique names.")

    num_cores = args.num_cores

    workspace_path = setup_workspace(args.output_dir)
    structures_dir = workspace_path / "structures"

    pair_info: dict[str, tuple[str, str, str]] = {}

    for pdb_file in pdb_files:
        # Copy PDB file to workspace
        copied_pdb_file = workspace_path / pdb_file.name
        shutil.copy(pdb_file, copied_pdb_file)

        pdb_input = PDBInput.from_file(copied_pdb_file).renumber()

        ## Generate embeddings (one ESM call per unique sequence) and materialize
        ## one .pt file per model/chain label for graph generation
        embeddings = pdb_input.gen_embeddings()
        pdb_input.write_embeddings(embeddings, output_dir=workspace_path)

        ## Materialize one 2-chain PDB file per chain pair of each model
        ## (the GNN scores one interface at a time), with matching embeddings
        pair_info.update(pdb_input.write_chain_pairs(structures_dir, workspace_path))

    if not pair_info:
        parser.error("No chain pairs found: every input PDB has a single chain.")

    num_cores = min(num_cores, MAX_cores, len(pair_info))
    log.info(f"Using {num_cores} cores for processing")

    ## Generate graph (one hdf5 covering every model of every input)
    graph = create_graph(
        pdb_path=structures_dir, workspace_path=workspace_path, nproc=num_cores
    )
    ## Predict fnat
    csv_output = predict(
        input_info=graph, workspace_path=workspace_path, ncores=num_cores
    )

    ## Present the results
    parse_output(
        csv_output=csv_output,
        pair_info=pair_info,
    )

    # Save the final prediction where the user invoked the command
    result_file = Path.cwd() / Path(csv_output).name
    shutil.copy(csv_output, result_file)
    log.info(f"Result saved to {result_file}")


if __name__ == "__main__":
    main()
