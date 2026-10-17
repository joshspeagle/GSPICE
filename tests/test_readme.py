import re
import runpy
from pathlib import Path

import pytest

README = Path(__file__).parents[1] / "README.md"


@pytest.mark.filterwarnings("error")
def test_the_readme_example_runs(tmp_path, monkeypatch):
    # The first Python block, as a script in an empty directory: it writes model.npz.
    text = README.read_text(encoding="utf-8")
    script = tmp_path / "example.py"
    script.write_text(re.search(r"```python\n(.*?)```", text, re.DOTALL)[1], "utf-8")
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script))
