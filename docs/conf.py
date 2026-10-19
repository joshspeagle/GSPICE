"""Sphinx configuration for the GSPICE documentation."""

import gspice

project = "GSPICE"
author = "Douglas P. Finkbeiner, Joshua S. Speagle and Tanveer Karim"
copyright = "2021-2026, the GSPICE developers"
release = gspice.__version__

extensions = ["myst_parser", "sphinx_copybutton", "sphinxcontrib.bibtex"]
# The design document is for developers, and the repository links it.
exclude_patterns = ["_build", "design.md"]
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

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
