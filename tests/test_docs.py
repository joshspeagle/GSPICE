import re
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

import gspice

ROOT = Path(__file__).parents[1] / "docs"
DOCS = ROOT / "src" / "content" / "docs"
# install.md runs shell commands, and results.md shows a call without its inputs.
NOT_SCRIPTS = {"install", "results"}
PAGES = [
    *sorted(p for p in (DOCS / "start").glob("*.md*") if p.stem not in NOT_SCRIPTS),
    *sorted((DOCS / "guides").glob("*.md*")),
    *sorted((DOCS / "legacy").glob("*.md*")),
    DOCS / "example" / "survey.mdx",
]
# The worked example reads the survey of docs/scripts/spectra.py, a download. Its code
# runs here on a small stand-in, whose numbers differ from its output blocks;
# docs/scripts/survey.py checks those on the survey itself.
STAND_IN = {"survey"}
# A fence with the flag output holds the output of the code above it.
OUTPUT = re.compile(r"```text output\n(.*?)```", re.DOTALL)


@pytest.fixture(scope="module")
def survey(tmp_path_factory):
    # A stand-in for spectra.npz with the keys that the worked example reads: 5000
    # spectra, enough for fit_blocks on a fifth of them and for snr_bins=3 on 60% of
    # them, on 400 pixels over the survey's wavelength range, of which 160 lie in the
    # range that the page fits.
    nspec, npix = 5000, 400
    sim = gspice.simulate(nspec, npix, rng=np.random.default_rng(121), artifacts=0.9)
    # The noise-free flux stands in for the flux without artifacts at their pixels.
    clean = np.where(sim.artifacts, sim.truth, sim.flux)
    # Each spectrum has at most one artifact, so its size gives its type.
    size = sim.artifacts.sum(axis=1, keepdims=True)
    path = tmp_path_factory.mktemp("survey") / "spectra.npz"
    np.savez(
        path,
        wave=np.geomspace(3802.0, 8995.0, npix),
        flux=sim.flux.astype(np.float32),
        ivar=sim.ivar.astype(np.float32),
        clean=clean.astype(np.float32),
        clean_ivar=sim.ivar.astype(np.float32),
        cosmic=sim.artifacts & (size <= 3),
        column=sim.artifacts & (size > 3) & (size <= 20),
        shift=sim.artifacts & (size > 20),
        snr=sim.snr.astype(np.float32),
    )
    return path


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.stem)
def test_the_page_examples_run(page, tmp_path, monkeypatch, capsys, request):
    # The Python blocks of the page, in order, as one script in a directory of its own.
    # The migration page catches the DeprecationWarning that it shows. A page with output
    # blocks shows the output of its code there, except on a stand-in.
    text = page.read_text(encoding="utf-8")
    if page.stem in STAND_IN:
        shutil.copy(request.getfixturevalue("survey"), tmp_path)
    script = tmp_path / f"{page.stem}.py"
    script.write_text(
        "".join(re.findall(r"```python\n(.*?)```", text, re.DOTALL)), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script))
    if page.stem not in STAND_IN and (shown := OUTPUT.findall(text)):
        assert capsys.readouterr().out == "".join(shown)


def test_the_install_page_shows_the_version():
    text = (DOCS / "start" / "install.md").read_text(encoding="utf-8")
    assert OUTPUT.findall(text) == [f"{gspice.__version__}\n"]


def test_the_cite_page_copies_the_bibtex_entry_of_the_paper():
    def normal(text):
        return " ".join(text.split())

    page = (ROOT / "reference" / "cite.md").read_text(encoding="utf-8")
    bib = (ROOT / "references.bib").read_text(encoding="utf-8")
    block = re.search(r"```bibtex\n(.*?)```", page, re.DOTALL)[1]
    entry = re.search(r"@\w+\{finkbeiner2025gspice,.*?\n\}", bib, re.DOTALL)[0]
    assert normal(block) == normal(entry)
