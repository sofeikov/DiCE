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
uv sync
uv sync --extra deeplearning
uv sync --group test
uv sync --group lint
uv sync --extra deeplearning --group test --group lint
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
uv run isort . -c
uv run isort .
uv run flake8 . --max-complexity=30
uv run flake8 dice_ml/data_interfaces/ --max-complexity=10
```

`.flake8` sets `max-line-length = 127` and excludes `.venv`, `docs`, `build`, and `dist`.

## Architecture

### Entry Points

Users interact through three top-level classes exposed via `dice_ml/__init__.py`:

1. `dice_ml.Data` wraps dataset metadata such as feature ranges, categorical levels, and the outcome name. It dispatches to `PublicData` or `PrivateData` via runtime `__class__` reassignment in `decide_implementation_type`.
2. `dice_ml.Model` wraps a trained ML model. It dispatches to backend-specific implementations such as `BaseModel` for sklearn and `PyTorchModel` via the same reassignment pattern.
3. `dice_ml.Dice` is the explainer entry point. It takes `Data` and `Model`, then dispatches to a method-specific explainer class based on the `method` parameter and model backend.

### Runtime Class Reassignment Pattern

All three top-level classes use the same pattern: `__init__` calls `decide_implementation_type`, which sets `self.__class__` to a concrete subclass and re-invokes `__init__`. This means the class you construct is not necessarily the class you get back. `Dice(...)` may return `DiceRandom`, `DiceGenetic`, `DiceKD`, and so on.

### Explainer Hierarchy

`ExplainerBase` in `explainer_interfaces/explainer_base.py` defines:

- `generate_counterfactuals()`
- `local_feature_importance()`
- `global_feature_importance()`
- `feature_importance()`

Concrete explainers in `explainer_interfaces/`:

- Model-agnostic: `DiceRandom`, `DiceGenetic`, `DiceKD`
- Gradient-based: `DicePyTorch`
- Feasibility: `FeasibleBaseVAE`, `FeasibleModelApprox`

### Method Routing

`dice.py:decide()` maps method strings to explainer classes:

- `"random"` -> `DiceRandom`
- `"genetic"` -> `DiceGenetic`
- `"kdtree"` -> `DiceKD`
- `"gradient"` -> `DicePyTorch`

### Data Interfaces

- `PublicData` in `public_data_interface.py` has dataframe access, computes stats such as MAD and ranges, and handles one-hot encoding.
- `PrivateData` in `private_data_interface.py` is metadata-only and does not expose raw data.
- `kdtree` is incompatible with `PrivateData` because it needs training data for the tree.

### Output Format

- `CounterfactualExamples` in `diverse_counterfactuals.py` stores counterfactuals for one query instance.
- `CounterfactualExplanations` in `counterfactual_explanations.py` stores counterfactuals for multiple query instances and feature importance scores. It supports JSON serialization with schemas in `dice_ml/schema/`.

### Constants

`dice_ml/constants.py` defines enums such as:

- `BackEndTypes` (`sklearn`, `PYT`)
- `SamplingStrategy` (`random`, `genetic`, `kdtree`, `gradient`)
- `ModelTypes` (`classifier`, `regressor`)

## CI

GitHub Actions workflows run on Python 3.12-3.13 across ubuntu, macOS, and Windows using `uv`:

- `python-package.yml`
- `python-linting.yml`
- `notebook-tests.yml`
- `python-package-conda.yml`
