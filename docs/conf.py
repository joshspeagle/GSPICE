"""Sphinx configuration for the GSPICE documentation."""

import re

import gspice

project = "GSPICE"
author = "Douglas P. Finkbeiner, Joshua S. Speagle and Tanveer Karim"
copyright = "2021-2026, the GSPICE developers"
release = gspice.__version__

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]
nitpicky = True
napoleon_google_docstring = False
# The type slot holds shapes and sizes ("ndarray, shape (nspec, npix)"), which
# cross-references cannot resolve.
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_use_ivar = False
default_role = "py:obj"
# Numpy types that intersphinx does not resolve, the private types of the signatures,
# and the words of the type slot of an Attributes section, which napoleon links as
# classes: "ndarray, shape (nspec, npix)".
nitpick_ignore_regex = [
    (
        "py:class",
        (
            r"numpy\.(float64|int64)|numpy\..*\.(NDArray|ArrayLike)"
            r"|gspice\._predict\._Covariances"
            r"|ndarray|int64|shape|each shape|nspec|npix|nbin|nblock|k|K"
        ),
    )
]
autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
# The design document is for developers, and the repository links it.
exclude_patterns = ["_build", "design.md"]
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"


def escape_bars(app, what, name, obj, options, lines):
    # The docstrings write the size of a set as |G|, which reST reads as a substitution.
    # Bars stay in literal blocks (the lines after one that ends in "::", indented
    # deeper than it), in doctest lines, and in inline literals and roles.
    block = None
    for i, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        if block is not None and (not line.strip() or indent > block):
            continue
        text = line.lstrip()
        opens = line.endswith("::") and not text.startswith("..")
        block = indent if opens else None
        if text.startswith((">>>", "...")):
            continue
        parts = re.split(r"(`+[^`]*`+)", line)
        lines[i] = "".join(
            p if j % 2 else p.replace("|", r"\|") for j, p in enumerate(parts)
        )


def setup(app):
    app.connect("autodoc-process-docstring", escape_bars)


html_theme = "furo"
html_title = "GSPICE"
html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "light_logo": "logo-light.svg",
    "dark_logo": "logo-dark.svg",
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/joshspeagle/GSPICE/",
    "source_branch": "main",
    "source_directory": "docs/",
}
