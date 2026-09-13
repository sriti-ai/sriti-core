# Contributing to Sriti Core

Thank you for your interest in contributing. This guide covers everything you need to get started.

## Prerequisites

- Python 3.11+
- A Redis-compatible store running locally — [Valkey](https://valkey.io/) (recommended) or Redis 7+
- Git

## Setting Up

```bash
git clone https://github.com/sriti-ai/sriti-core.git
cd sriti-core

# Using a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Or with conda
conda create -n sriti python=3.11
conda activate sriti
pip install -e ".[dev]"
```

## Running Tests

```bash
# Unit tests only (no external services required)
pytest tests/ -q

# Include integration tests (requires Valkey/Redis and LLM API keys in .env)
pytest tests/ -q --run-integration
```

All tests must pass before a PR can be merged. The CI gate runs the unit suite only.

## Code Style

- Follow the conventions already in the codebase — consistent naming, docstrings on public functions, type hints throughout.
- Do not introduce new runtime dependencies without opening a discussion issue first. The dependency footprint is deliberately kept lean.
- No reformatting passes on unrelated code in a PR.

## Branch Naming

| Type | Prefix | Example |
|------|--------|---------|
| New feature | `feature/` | `feature/streaming-cache` |
| Bug fix | `fix/` | `fix/cold-start-tie-break` |
| Documentation | `docs/` | `docs/policy-yaml-reference` |
| Refactor | `refactor/` | `refactor/engine-dispatch` |

## Pull Request Process

1. For non-trivial changes, open a GitHub Issue first to discuss the approach.
2. Keep PRs focused — one logical change per PR.
3. Write or update tests for any changed behaviour.
4. Ensure `pytest tests/ -q` passes locally before opening the PR.
5. Fill in the PR description — what changed and why.

## Reporting Bugs

Open a [GitHub Issue](https://github.com/sriti-ai/sriti-core/issues) with:
- A minimal reproduction case
- The Python version and OS
- The full traceback if applicable

## Requesting Features

Open a GitHub Issue tagged `enhancement`. Describe the use case, not just the feature — it helps frame the discussion.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
