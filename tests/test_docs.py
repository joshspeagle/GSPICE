import re
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

import gspice

DOCS = Path(__file__).parents[1] / "docs" / "src" / "content" / "docs"
# install.md runs shell commands, and results.md shows a call without its inputs.
NOT_SCRIPTS = {"install", "results"}
PAGES = [
    *sorted(p for p in (DOCS / "start").glob("*.md*") if p.stem not in NOT_SCRIPTS),
    *sorted((DOCS / "tutorials").glob("*.md*")),
]
# A fence with the flag output holds the output of the code above it.
OUTPUT = re.compile(r"```text output\n(.*?)```", re.DOTALL)


@pytest.fixture(scope="module")
def survey(tmp_path_factory):
    # A stand-in for the output of docs/scripts/spectra.py, with the keys that the pages
    # read: 5000 spectra, enough for fit_blocks on a fifth of them, on 400 pixels over
    # the survey's wavelength range, of which 160 lie in the range that the pages fit.
    nspec, npix = 5000, 400
    sim = gspice.simulate(nspec, npix, rng=np.random.default_rng(121), artifacts=0.9)
    # The noise-free flux stands in for the flux without artifacts at their pixels.
    clean = np.where(sim.artifacts, sim.truth, sim.flux)
    # Each spectrum has at most one artifact, so its size gives its type. A shifted
    # block covers 200 or more pixels; it keeps its first 40, so that every spectrum
    # keeps good pixels where the pages fit.
    size = sim.artifacts.sum(axis=1, keepdims=True)
    shifted = sim.artifacts & (size > 20)
    shift = shifted & (np.cumsum(shifted, axis=1) <= 40)
    flux = np.where(shifted & ~shift, clean, sim.flux)
    path = tmp_path_factory.mktemp("survey") / "spectra.npz"
    np.savez(
        path,
        wave=np.geomspace(3802.0, 8995.0, npix),
        flux=flux.astype(np.float32),
        ivar=sim.ivar.astype(np.float32),
        clean=clean.astype(np.float32),
        clean_ivar=sim.ivar.astype(np.float32),
        cosmic=sim.artifacts & (size <= 3),
        column=sim.artifacts & (size > 3) & (size <= 20),
        shift=shift,
    )
    return path


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.stem)
def test_the_page_examples_run(page, survey, tmp_path, monkeypatch, capsys):
    # The Python blocks of the page, in order, as one script in a directory that holds
    # spectra.npz. The migration page catches the DeprecationWarning that it shows. A
    # page with output blocks shows the output of its code there.
    text = page.read_text(encoding="utf-8")
    shutil.copy(survey, tmp_path)
    script = tmp_path / f"{page.stem}.py"
    script.write_text("".join(re.findall(r"```python\n(.*?)```", text, re.DOTALL)))
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script))
    if shown := OUTPUT.findall(text):
        assert capsys.readouterr().out == "".join(shown)


def test_the_install_page_shows_the_version():
    text = (DOCS / "start" / "install.md").read_text(encoding="utf-8")
    assert OUTPUT.findall(text) == [f"{gspice.__version__}\n"]
