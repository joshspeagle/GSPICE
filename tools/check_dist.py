"""Check that the sdist holds the sources and tests and the wheel holds the package only."""

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]


def tracked(*directories: str) -> set[str]:
    names = subprocess.run(
        ["git", "ls-files", *directories],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(names.split())


def check(dist: Path) -> list[str]:
    (wheel,) = dist.glob("*.whl")
    (sdist,) = dist.glob("*.tar.gz")
    problems = []

    with zipfile.ZipFile(wheel) as z:
        held = {n for n in z.namelist() if ".dist-info/" not in n}
    wanted = {p.removeprefix("src/") for p in tracked("src")}
    problems += [f"wheel lacks {p}" for p in sorted(wanted - held)]
    problems += [f"wheel holds {p}" for p in sorted(held - wanted)]

    with tarfile.open(sdist) as t:
        top = f"{sdist.name.removesuffix('.tar.gz')}/"
        held = {n.removeprefix(top) for n in t.getnames()}
    wanted = tracked("src", "tests")
    wanted |= {"pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE"}
    problems += [f"sdist lacks {p}" for p in sorted(wanted - held)]
    return problems


if __name__ == "__main__":
    found = check(Path(sys.argv[1]))
    print("\n".join(found) or "wheel and sdist hold the expected files")
    sys.exit(bool(found))
