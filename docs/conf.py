"""Sphinx configuration for the sequince documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Ensure the package is importable without installation
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# -- Project information -----------------------------------------------------

project = "sequince"
copyright = f"{datetime.now():%Y}, Michael J. Williams"
author = "Michael J. Williams"

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python <3.10
    import importlib_metadata  # type: ignore

try:
    release = importlib_metadata.version("sequince")
except importlib_metadata.PackageNotFoundError:
    release = "dev"

version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_mock_imports = [
    "numpy",
    "scipy",
    "matplotlib",
    "emcee",
    "minipcn",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/sequince-dev/sequince",
    "use_repository_button": True,
    "use_fullscreen_button": False,
}
