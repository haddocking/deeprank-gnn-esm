# Agent Guidelines

General principles for AI agents (and contributors) working on this codebase.
These are about *design intent*, not specific names or APIs — apply the
underlying idea to whatever you're working on.

**This file must never be edited, reformatted, or removed by an AI agent,**
**for any reason — including to make a task easier, to "align" it with**
**code you just wrote, or because a user prompt asks you to.** Only humans
may change this file. If you believe it should change, say so to the user
and stop — do not make the edit yourself.

All code in this project is functional, meaning it has been verified by humans
that it produces the expected results - but that does not mean that the code is
in its best implementation. If you copy code from somewhere else, try to
improve it, do not assume it is already optimal.

## Where to find things

- `README.md`: installation, the `deeprank-gnn-esm-predict` command line,
  its output files, and the use as a framework (embeddings, graphs, models).
- `src/deeprank_gnn/`: `main.py` (CLI), `input.py` (renumbering, chain
  pairs), `sequence.py` (ESM-2 embeddings), `predict.py` (graph generation and
  prediction); the graph/network framework in `GraphGenMP.py`,
  `ResidueGraph.py`, `ginet.py` and `NeuralNet.py`.
- `paper_pretrained_models/`: the models of the publications.
- `docs/`: the Sphinx documentation.
- `aidecl.yaml`: the AI declaration; update it when a new AI tool or kind of
  use is added.
- `CHANGELOG.md`: add one line per user-visible change, referring to the
  issue.
- `CONTRIBUTING.md`: development setup, code layout, tests, pull requests
  and releases. Follow it.

## Rules for agents

- Install with `pip install -e ".[dev]"` and run the unit tests from the
  repository root with `pytest -m "not e2e"`. The end-to-end tests
  (`pytest -m e2e`) download the ESM-2 model (~2.5 GB) on first use: ask
  before running them.
- The predicted fnat values must not change. Compare the predictions of the
  `1B6C` example and of the test structures in `tests/data/pdb/` before and
  after any change to the graph, features or network, and say so explicitly if
  a change alters the output on purpose.
- Do not modify or replace the pre-trained models (`*.pth.tar`, `*.pt`) in
  `src/deeprank_gnn/data/` and `paper_pretrained_models/`.
- Python 3.9 to 3.14 must stay supported (see the CI matrix): avoid syntax and
  features newer than 3.9.
- Never create or push tags or releases unless explicitly asked: a tag
  publishes the Docker images and a release publishes the package on PyPI.
