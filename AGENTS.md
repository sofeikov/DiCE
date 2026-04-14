# AGENTS.md

This file is the canonical repository guidance for coding agents working in this repository.
Tool-specific guidance files such as `CLAUDE.md` and `.github/copilot-instructions.md` should defer to this file instead of duplicating instructions.

## Repository Guidance

- For any user-visible behavior change, public API change, bug fix that affects documented behavior, or workflow change, explicitly check whether `README.rst`, source docs, examples, tests, and changelog or release notes need updates before concluding the task.
- If the repository has no changelog or release-notes file, say that explicitly in the final report instead of inventing one.
- Prefer updating source documentation over generated files in `docs/`.

## What is DiCE

DiCE (Diverse Counterfactual Explanations) is a Python library for generating counterfactual explanations for ML models. Given a model prediction, it answers "what minimal changes to the input would flip the outcome?" It supports sklearn and PyTorch backends, with multiple CF generation strategies (random sampling, genetic algorithm, KD-tree, gradient-based).

## Build & Install

```bash
uv sync                                        # core deps only
uv sync --extra deeplearning                   # include torch
uv sync --group test                           # include test deps
uv sync --group lint                           # include linting deps
uv sync --extra deeplearning --group test --group lint  # everything
```

## Testing

```bash
# Run all tests (excludes notebook tests)
uv run pytest tests/ -m "not notebook_tests" --doctest-modules

# Run a single test file
uv run pytest tests/test_dice_interface/test_dice_random.py

# Run a single test by name
uv run pytest tests/test_dice_interface/test_dice_random.py -k "test_name"

# Run notebook tests only
uv run pytest tests/ -m "notebook_tests"
```

Tests use session/module-scoped fixtures in `tests/conftest.py` that are parametrized over backends (`sklearn`, `PYT`) and data interfaces (`private`, `public`).

## Linting

```bash
uv run isort . -c                  # check import sorting
uv run isort .                     # fix import sorting
uv run flake8 . --max-complexity=30  # line length configured in .flake8
# data_interfaces has stricter complexity limit:
uv run flake8 dice_ml/data_interfaces/ --max-complexity=10
```

Flake8 config (`.flake8`) sets `max-line-length = 127` and excludes `.venv`, `docs`, `build`, `dist`.

## Architecture

### Entry Points (the three-step API)

Users interact through three top-level classes exposed via `dice_ml/__init__.py`:

1. **`dice_ml.Data`** — wraps dataset metadata (feature ranges, categorical levels, outcome name). Dispatches to `PublicData` (has a dataframe) or `PrivateData` (metadata-only) via runtime `__class__` reassignment in `decide_implementation_type`.

2. **`dice_ml.Model`** — wraps a trained ML model. Dispatches to backend-specific implementations (`BaseModel` for sklearn, `PyTorchModel`) via the same `__class__` reassignment pattern.

3. **`dice_ml.Dice`** — the explainer. Takes Data + Model, dispatches to a method-specific explainer class based on the `method` parameter and model backend.

### Runtime class reassignment pattern

All three top-level classes use the same pattern: the `__init__` calls `decide_implementation_type` which sets `self.__class__` to a concrete subclass and re-invokes `__init__`. This means the class you construct is not the class you get back — `Dice(...)` may return a `DiceRandom`, `DiceGenetic`, `DiceKD`, etc.

### Explainer hierarchy

`ExplainerBase` (ABC in `explainer_interfaces/explainer_base.py`) defines:
- `generate_counterfactuals()` — main public API, handles validation, delegates to `_generate_counterfactuals()`
- `local_feature_importance()` / `global_feature_importance()` / `feature_importance()`

Concrete explainers in `explainer_interfaces/`:
- **Model-agnostic**: `DiceRandom`, `DiceGenetic`, `DiceKD` — work with any backend
- **Gradient-based**: `DicePyTorch` — requires differentiable models
- **Feasibility**: `FeasibleBaseVAE`, `FeasibleModelApprox` — VAE-based feasible CF generation

### Method routing

`dice.py:decide()` maps method strings to explainer classes:
- `"random"` → `DiceRandom` (default for sklearn)
- `"genetic"` → `DiceGenetic`
- `"kdtree"` → `DiceKD` (CFs from training data only; requires public data)
- `"gradient"` → `DicePyTorch`

### Data interfaces

- `PublicData` (`public_data_interface.py`) — has full dataframe access, computes stats (MAD, ranges), handles one-hot encoding
- `PrivateData` (`private_data_interface.py`) — metadata only (feature names + ranges/levels), no raw data access
- KD-tree method is incompatible with `PrivateData` (needs training data for the tree)

### Output format

- `CounterfactualExamples` (`diverse_counterfactuals.py`) — stores CFs for a single query instance
- `CounterfactualExplanations` (`counterfactual_explanations.py`) — stores CFs for multiple query instances + feature importance scores. Supports JSON serialization with schema validation (v1.0 and v2.0 schemas in `dice_ml/schema/`).

### Constants

`dice_ml/constants.py` defines enums: `BackEndTypes` (`sklearn`, `PYT`), `SamplingStrategy` (`random`, `genetic`, `kdtree`, `gradient`), `ModelTypes` (`classifier`, `regressor`).

## CI

GitHub Actions workflows run on Python 3.12-3.13 across ubuntu/macos/windows using `uv`:
- `python-package.yml` — full test suite
- `python-linting.yml` — flake8 + isort
- `notebook-tests.yml` — Jupyter notebook integration tests
- `python-package-conda.yml` — conda environment test
