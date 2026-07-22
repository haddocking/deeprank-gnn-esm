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
from deeprank_gnn.sequence import MultiFasta


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

    is_temp_workspace = args.output_dir is None
    workspace_path = setup_workspace(args.output_dir)
    structures_dir = workspace_path / "structures"

    try:
        pdb_inputs = []
        combined_fasta = MultiFasta()

        for pdb_file in pdb_files:
            # Copy PDB file to workspace
            copied_pdb_file = workspace_path / pdb_file.name
            if copied_pdb_file.resolve() != pdb_file.resolve():
                shutil.copy(pdb_file, copied_pdb_file)

            pdb_input = PDBInput.from_file(copied_pdb_file).renumber()

            ## Collect this input's unique sequences into the shared pool, so
            ## the ESM model is loaded and run once for every input PDB combined
            for seq in pdb_input.to_multi_fasta().sequences.values():
                combined_fasta.add(seq)

            pdb_inputs.append(pdb_input)

        ## Generate embeddings for every unique sequence across all inputs
        ## (one ESM model load, one batch pass) and materialize one .pt file
        ## per model/chain label for graph generation
        embeddings = combined_fasta.gen_embeddings()

        pair_info: dict[str, tuple[str, str, str]] = {}

        for pdb_input in pdb_inputs:
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
    finally:
        if is_temp_workspace:
            shutil.rmtree(workspace_path, ignore_errors=True)


if __name__ == "__main__":
    main()
