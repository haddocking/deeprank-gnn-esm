# Contributing to deeprank-gnn-esm

Bug reports, questions and pull requests are welcome; use the issue templates
on GitHub. This file describes how to set up a development environment, check
your changes and get them merged. For using deeprank-gnn-esm, see `README.md`.

## Development setup

deeprank-gnn-esm supports Python 3.9 to 3.14. Your Python must be compiled with
`sqlite3` support (see `README.md`).

```bash
git clone https://github.com/haddocking/deeprank-gnn-esm.git
cd deeprank-gnn-esm
python -m venv .venv
source .venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu  # optional, CPU-only torch
pip install -e '.[dev]'  # with pytest, pytest-cov and hypothesis
```

The package metadata and dependencies are in `pyproject.toml`. This is an
editable install: changes to the code in `src/deeprank_gnn/` take effect
without reinstalling.

## Code layout

The package is in `src/deeprank_gnn/`:

- `main.py`: the `deeprank-gnn-esm-predict` command line.
- `input.py`: the `PDBInput` class; renumbers the input structures, splits
  ensembles into models and writes a two-chain PDB file for every chain pair.
- `sequence.py`: the `Sequence` and `MultiFasta` classes; computes the ESM-2
  embeddings of the unique chain sequences.
- `predict.py`: sets up the workspace, generates the graphs, runs the
  pre-trained model and writes the CSV output.
- `logger.py`: the logger.
- `GraphGenMP.py`, `ResidueGraph.py`, `Graph.py`: generation of the residue
  graphs and their features, stored in HDF5 files.
- `DataSet.py`, `NeuralNet.py`, `ginet.py`, `community_pooling.py`,
  `Metrics.py`: the graph neural network, its training, evaluation and metrics.
- `tools/`: buried surface area, PSSM, embedding and HDF5 utilities.
- `data/`: the pre-trained model used by `deeprank-gnn-esm-predict`.

Outside the package: `tests/` (unit and end-to-end tests, with their data in
`tests/data/`), `scripts/` (PDB renumbering and FASTA extraction),
`paper_pretrained_models/` (the models of the publications) and `docs/` (the
Sphinx documentation).

The pre-trained models (`*.pth.tar`) must not be changed or replaced.

## Code style

Follow the style of the surrounding code and [PEP 8](https://peps.python.org/pep-0008/).
Give new modules, public classes and functions a docstring (PEP 257, summary on
the first line) and type annotations. Do not use syntax or features newer than
Python 3.9.

## Tests

Run the tests from the repository root:

```bash
pytest -m "not e2e"            # unit tests
pytest -m "not e2e and not gpu"  # without the tests that need a CUDA GPU
pytest -m e2e                  # end-to-end tests of the full pipeline
```

The end-to-end tests run `deeprank-gnn-esm-predict` and download the ESM-2
model (`esm2_t33_650M_UR50D`, ~2.5 GB) into `~/.cache/torch/hub/checkpoints`
on first use.

The predicted fnat values must not change. After any change to the graph
generation, the features or the network, compare the predictions of the
`1B6C` example of `README.md` and of the structures in `tests/data/pdb/` before
and after the change. If a change intentionally alters the predictions,
explain why in the pull request.

## Pull requests

Work on a branch and open a pull request into `main`; fill in the pull-request
template. Before it can be merged, a pull request needs:

- all review conversations resolved,
- a branch that is up to date with `main`.

The GitHub workflow (`.github/workflows/ci.yml`) runs on every pull request:
the unit tests on Linux (x86_64 and ARM) and macOS with Python 3.9 and 3.14,
the end-to-end tests on Linux with Python 3.14, and the build of the CPU and
GPU Docker images. Run the tests locally before pushing.
These jobs are skipped when a pull request changes only `*.md` files, or when
a push does not change the code of the pull request (e.g. merging `main`) and
`ci-passed` succeeded on the previous commit; `ci-passed` then still reports
success.

If AI tools played a meaningful role, say so in the pull request (the template
has a section for it), and update `aidecl.yaml`, the project's AI declaration,
when a new tool or kind of use is added.

Checklist:

- Tests added for new code.
- Documentation added: docstrings, and `README.md` / `docs/` when the usage
  or installation changes.
- `CHANGELOG.md` updated: one line per user-visible change, referring to the
  issue.
- Predicted fnat values unchanged, or changed with an explanation.
- No new dependencies without discussion.
- Prefer small, incremental changes, with passing tests at each step, over
  large rewrites.

## Releases

The version is set in `pyproject.toml` only; `deeprank_gnn.__version__` reads
it from the installed package metadata and `docs/conf.py` from
`pyproject.toml`. To release, update the version and `CHANGELOG.md`, merge,
then create a GitHub release with a tag on `main`. Publishing the release runs
`.github/workflows/publish.yml`, which publishes the package on PyPI, and
pushing the tag runs `.github/workflows/docker-publish.yml`, which publishes
the CPU and GPU Docker images to ghcr.io.
