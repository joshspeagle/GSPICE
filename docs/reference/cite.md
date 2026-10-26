---
title: Cite GSPICE
description: The paper to cite, the citation file of the software, and the data of the worked example.
editUrl: https://github.com/joshspeagle/GSPICE/edit/main/docs/reference/cite.md
---

If you use GSPICE, cite the paper that describes the method [@finkbeiner2025gspice]:

```bibtex
@article{finkbeiner2025gspice,
  author  = {Finkbeiner, Douglas P. and Speagle, Joshua S. and Karim, Tanveer},
  title   = {Data-Driven Stellar Spectral Modelling with {GSPICE}},
  journal = {arXiv e-prints},
  year    = {2025},
  eprint  = {2511.16754},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2511.16754},
}
```

## The software

[`CITATION.cff`](https://github.com/joshspeagle/GSPICE/blob/main/CITATION.cff) holds the
citation metadata of the software: its authors, version, licence and release date. It
names the paper above as the preferred citation. On GitHub, "Cite this repository" in
the sidebar of the repository gives the same metadata as APA and BibTeX. To cite the
version that you used, give the value of `gspice.__version__`.

## The worked example

[A survey end to end](/GSPICE/example/survey/) simulates its spectra from two data sets.
If you use these spectra, cite both:

- The PHOENIX stellar models, the library PHOENIX-ACES-AGSS-COND-2011
  [@husser2013phoenix]. The library asks its users to cite this paper.
- The sky emission lines of the UVES sky atlas [@hanuschik2003uves].

[References](/GSPICE/references/) lists every work that the site cites.
