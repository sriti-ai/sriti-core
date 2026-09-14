# Changelog

## 0.1.2 (2026-09-14)

- Include `sriti/config/*.yaml` files in the published wheel via `[tool.setuptools.package-data]` and `MANIFEST.in`; fixes `FileNotFoundError` on startup after `pip install sriti-core`.
- Policy loading (`sriti/core/cascade/policy.py`) now falls back to sane built-in defaults with a warning instead of crashing with an unhandled exception when `policy.yaml` cannot be opened.

## 0.1.1 (2024-12-xx)

- Fix: remove `numpy<2` upper bound to resolve fastembed compatibility.

## 0.1.0 (2024-12-xx)

- Initial release of sriti-core.
