---
title: Install
description: Install GSPICE from PyPI, and check that Python can import it.
---

This page installs GSPICE and checks that Python can import it.

## Requirements

GSPICE needs Python 3.11 or later, numpy 1.26 or later and scipy 1.11 or later. pip
installs numpy and scipy if they are missing.

## Install from PyPI

GSPICE is on the Python Package Index (PyPI) as `pygspice`. Its import name is
`gspice`:

```sh
pip install pygspice
```

An install from GitHub made before 2026-10-29 is named `gspice`. If `pip show gspice`
finds it, run `pip uninstall gspice` first, because both packages write the same files.

## Install from a clone

A clone of the repository also holds the tests and the scripts of these pages. Clone
it, then install it from its folder:

```sh
git clone https://github.com/joshspeagle/GSPICE
cd GSPICE
pip install .
```

## Check the installation

Print the version of GSPICE:

```sh
python -c "import gspice; print(gspice.__version__)"
```

```text output
2.0.0
```

If Python prints `ModuleNotFoundError: No module named 'gspice'`, pip installed GSPICE
for another Python. Run `python -m pip install` instead of `pip install`, with the
`python` that you use.
