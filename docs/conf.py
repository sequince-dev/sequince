"""Sphinx configuration for the sequince documentation."""

from __future__ import annotations

from datetime import datetime

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
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Generate module pages from source without importing optional backends.
autoapi_dirs = ["../src/sequince"]
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
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
