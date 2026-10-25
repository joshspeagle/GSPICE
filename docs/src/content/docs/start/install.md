---
title: Install
description: Install GSPICE from GitHub, and check that Python can import it.
---

This page installs GSPICE and checks that Python can import it.

## Requirements

GSPICE needs Python 3.11 or later, numpy 1.26 or later and scipy 1.11 or later. pip
installs numpy and scipy if they are missing.

## Install from GitHub

GSPICE is not on the Python Package Index (PyPI) yet. Install it from its GitHub
repository, which needs `git`:

```sh
pip install git+https://github.com/joshspeagle/GSPICE
```

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
