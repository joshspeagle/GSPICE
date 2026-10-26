"""Write the API reference pages of the site from the docstrings of src/gspice.

    python docs/scripts/api.py

Each page of docs/reference/ is a template: a line "::: name" becomes the reference of
the object name, which griffe reads from src/gspice without importing the package. The
pages go to docs/src/content/docs/reference/, which the Astro build reads.

The script fails, naming the object, on a griffe warning, on a docstring section that
it does not render, and on a backtick reference (a cross-reference, as with a default
role in reST) that names neither a documented object nor an object of the objects.inv
of Python, NumPy or SciPy.
It also fails unless the templates document each name of gspice.__all__ and of
gspice.legacy.__all__ once. It downloads the three inventories at each run.
"""

import logging
import logging.handlers
import re
import shutil
import sys
import textwrap
import urllib.request
import zlib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import griffe

DOCS = Path(__file__).parents[1]
SRC = DOCS.parent / "src"
TEMPLATES = DOCS / "reference"
OUT = DOCS / "src" / "content" / "docs" / "reference"
INVENTORIES = (
    "https://docs.python.org/3/",
    "https://numpy.org/doc/stable/",
    "https://docs.scipy.org/doc/scipy/",
)
# The kinds that griffe gives the NumPy sections of name and description items; the
# title of a section is its kind in title case.
FIELDS = {"parameters", "other parameters", "attributes", "returns", "raises"}


def plain(text: str) -> str:
    # Markdown syntax that is plain text in reST; the escaped | also keeps a | in prose,
    # such as in the legacy docstrings, from opening a table cell.
    return re.sub(r"([\\*_|<$~])", r"\\\1", text)


def inventory(base: str) -> Iterator[tuple[str, str]]:
    # objects.inv: four header lines, then zlib-compressed rows
    # "name domain:role priority uri dispname", where "$" in uri stands for name.
    # numpy.org refuses the default User-Agent of urllib.
    request = urllib.request.Request(
        base + "objects.inv", headers={"User-Agent": "gspice-docs"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        version = response.readline()
        if version != b"# Sphinx inventory version 2\n":
            raise SystemExit(f"{base}objects.inv: not a Sphinx inventory of version 2")
        for _ in range(3):
            response.readline()
        rows = zlib.decompress(response.read()).decode()
    for row in rows.splitlines():
        if match := re.match(r"(\S+)\s+py:\w+\s+-?\d+\s+(\S+)", row):
            name, uri = match.groups()
            yield name, base + uri.replace("$", name)


def unalias(obj: griffe.Object | griffe.Alias) -> griffe.Object:
    return obj.final_target if obj.is_alias else obj


def documented_members(obj: griffe.Object) -> list[griffe.Object]:
    # As with the :members: option of autodoc: public members with a docstring, in
    # source order.
    return [
        unalias(member)
        for name, member in obj.members.items()
        if not name.startswith("_") and member.docstring
    ]


def scope(name: str) -> list[str]:
    # A reference resolves in the object, its class, then its module.
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


class Reference:
    def __init__(self, package: griffe.Module, pages: dict[str, list[str]]) -> None:
        self.package = package
        self.anchors: dict[str, str] = {}
        for page, names in pages.items():
            for name in names:
                obj = self.get(name)
                members = [member.name for member in documented_members(obj)]
                if obj.docstring:
                    members += [
                        item.name
                        for section in obj.docstring.parsed
                        if section.kind.value == "attributes"
                        for item in section.value
                    ]
                for target in [name, *(f"{name}.{member}" for member in members)]:
                    self.anchors[target] = f"/GSPICE/reference/{page}/#{target}"
        self.external: dict[str, str] = {}
        for base in INVENTORIES:
            for name, uri in inventory(base):
                self.external.setdefault(name, uri)

    def get(self, name: str) -> griffe.Object:
        return unalias(self.package[name.removeprefix("gspice.")])

    def link(self, target: str, name: str) -> str:
        for prefix in scope(name):
            if (anchor := self.anchors.get(f"{prefix}.{target}")) is not None:
                return anchor
        if (uri := self.anchors.get(target, self.external.get(target))) is not None:
            return uri
        raise SystemExit(f"{name}: unresolved reference `{target}`")


def text(line: str, reference: Reference, name: str) -> str:
    # A single backtick span is a cross-reference, a double one is code, and a :math:
    # role is inline maths. A span that wraps onto the next line would pass unchecked.
    if re.sub(r"``[^`]+``", "", line).count("`") % 2:
        raise SystemExit(f"{name}: a backtick span wraps onto the next line: {line!r}")
    out = []
    for i, part in enumerate(re.split(r"(:math:`[^`]+`|``[^`]+``|`[^`]+`)", line)):
        if i % 2 == 0:
            out.append(plain(part))
        elif part.startswith(":math:"):
            out.append(f"${part.removeprefix(':math:')[1:-1]}$")
        elif part.startswith("``"):
            out.append(part)
        else:
            out.append(f"[{part}]({reference.link(part[1:-1], name)})")
    return re.sub(r"^(\s*)([>#])", r"\1\\\2", "".join(out))


def prose(value: str, reference: Reference, name: str) -> list[str]:
    # A line ".. math::" opens a block of display maths, and another line that ends in
    # "::" a reST literal block: the lines after it that are blank or indented deeper
    # than it.
    out: list[str] = []
    lines = value.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.endswith("::"):
            out.append(text(line, reference, name))
            continue
        math = line.strip() == ".. math::"
        if not math:
            out.append(text(line[:-1], reference, name))
        indent = len(line) - len(line.lstrip())
        block = []
        while i < len(lines) and (
            not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent
        ):
            block.append(lines[i])
            i += 1
        if not "".join(block).strip():
            raise SystemExit(f"{name}: no literal block after {line!r}")
        # The block keeps the indent of its opening line, as in a list item.
        opening, closing = ("$$", "$$") if math else ("```text", "```")
        body = textwrap.dedent("\n".join(block)).strip("\n").split("\n")
        out += [
            "",
            *((line[:indent] + x) if x else x for x in [opening, *body, closing]),
        ]
        if i < len(lines):
            out.append("")
    return out


def headers(docstring: str, title: str, name: str) -> list[str]:
    # griffe rewrites a type that parses as Python ("array_like, shape (n, m)" becomes a
    # tuple with a call) and splits "flux, ivar : type" into one item per name, so the
    # pages take the item lines of a section from the docstring.
    lines = docstring.splitlines()
    starts = [
        i for i in range(len(lines)) if lines[i : i + 2] == [title, "-" * len(title)]
    ]
    if len(starts) != 1:
        raise SystemExit(f"{name}: no single section {title} in the docstring")
    start = starts[0]
    out = []
    for line in lines[start + 2 :]:
        if re.fullmatch(r"-+", line):
            return out[:-1]
        if line and not line[0].isspace():
            out.append(line)
    return out


def fields(
    obj: griffe.Object,
    name: str,
    section: griffe.DocstringSection,
    reference: Reference,
) -> list[str]:
    kind = section.kind.value
    title = kind.title()
    out = [f"**{title}:**", ""]
    items = list(section.value)
    for header in headers(obj.docstring.value, title, name):
        # An item line is "names : type", or "type" in Raises and Returns.
        names, _, type_text = header.rpartition(" : ")
        if kind == "raises":
            head, split = f"[**{header}**]({reference.link(header, name)})", [""]
        elif kind == "returns":
            head = f"*{plain(type_text)}*"
            head = f"**{names}** ({head})" if names else head
            split = [names]
        else:
            if not names:
                names, type_text = header, ""
            split = [n.strip() for n in names.split(",")]
            head = f"**{names}**" + (f" (*{plain(type_text)}*)" if type_text else "")
            if kind == "attributes":
                head = "".join(f'<span id="{name}.{n}"></span>' for n in split) + head
        group, items = items[: len(split)], items[len(split) :]
        if kind != "raises" and [item.name for item in group] != split:
            raise SystemExit(f"{name}: griffe and the docstring disagree on {header!r}")
        first, *rest = prose(group[0].description, reference, name) or [""]
        out += [f"- {head} – {first}", *(f"  {line}" if line else "" for line in rest)]
    if items:
        raise SystemExit(f"{name}: an item of {title} has no line of its own")
    return [*out, ""]


def annotation(expression: str | griffe.Expr, package: griffe.Module) -> str:
    # Show the value of a type alias of gspice: Integer is int | np.integer[Any].
    if not isinstance(expression, griffe.Expr):
        return str(expression)
    out = []
    for part in expression.iterate(flat=True):
        if isinstance(part, griffe.ExprName) and part.canonical_path.startswith(
            "gspice._"
        ):
            alias = package[part.canonical_path.removeprefix("gspice.")]
            if alias.is_attribute:
                out.append(str(alias.value))
                continue
        out.append(str(part))
    return "".join(out)


def signature(obj: griffe.Object, call: str, package: griffe.Module) -> str:
    params = [p for p in obj.parameters if p.name != "self"]
    kinds = [p.kind.value for p in params]
    out = []
    for i, p in enumerate(params):
        if kinds[i] == "keyword-only" and (
            i == 0 or kinds[i - 1] not in ("keyword-only", "variadic positional")
        ):
            out.append("*")
        text = {"variadic positional": "*", "variadic keyword": "**"}.get(
            kinds[i], ""
        ) + p.name
        if p.annotation is not None:
            text += f": {annotation(p.annotation, package)}"
        if p.default is not None:
            text += f" = {p.default}" if p.annotation is not None else f"={p.default}"
        out.append(text)
        if kinds[i] == "positional-only" and "positional-only" not in kinds[i + 1 :]:
            out.append("/")
    returns = (
        f" -> {annotation(obj.returns, package)}"
        if obj.is_function and obj.returns
        else ""
    )
    line = f"{call}({', '.join(out)}){returns}"
    if len(line) > 88:
        line = f"{call}(\n" + "".join(f"    {p},\n" for p in out) + f"){returns}"
    return line


def render(name: str, level: int, reference: Reference) -> list[str]:
    obj = reference.get(name)
    if not (obj.is_class or obj.is_function) or "property" in obj.labels:
        raise SystemExit(f"{name}: the pages render only classes and functions")
    call = name if level == 3 else name.rsplit(".", 1)[1]
    out = [f"{'#' * level} {call} {{#{name}}}", ""]
    # As autodoc does, a function with overloads shows their signatures.
    callables = obj.overloads if obj.is_function and obj.overloads else [obj]
    prefix = "class " if obj.is_class else ""
    out += [
        "```python",
        *(signature(c, prefix + call, reference.package) for c in callables),
        "```",
        "",
    ]
    for section in obj.docstring.parsed if obj.docstring else []:
        kind = section.kind.value
        if kind == "text":
            out += [*prose(section.value, reference, name), ""]
        elif kind in FIELDS:
            out += fields(obj, name, section, reference)
        elif kind == "admonition" and section.title == "Notes":
            out += [
                "**Notes**",
                "",
                *prose(section.value.contents, reference, name),
                "",
            ]
        else:
            raise SystemExit(
                f"{name}: unhandled docstring section {section.title or kind}"
            )
    for member in documented_members(obj):
        out += render(f"{name}.{member.name}", level + 1, reference)
    return out


def public(package: griffe.Module) -> Counter[str]:
    # The names of __all__, without the modules that it exports.
    names: Counter[str] = Counter()
    for module in (package, package["legacy"]):
        for name in module.exports:
            if not module[name].is_module:
                names[f"{module.path}.{name}"] += 1
    return names


def main() -> None:
    warnings = logging.handlers.BufferingHandler(sys.maxsize)
    warnings.setLevel(logging.WARNING)
    logging.getLogger("griffe").addHandler(warnings)

    package = griffe.load(
        "gspice", search_paths=[SRC], docstring_parser="numpy", resolve_aliases=True
    )
    templates = {
        path.stem: path.read_text(encoding="utf-8") for path in TEMPLATES.glob("*.md")
    }
    pages = {
        page: re.findall(r"^::: (\S+)$", source, re.MULTILINE)
        for page, source in templates.items()
    }
    documented = Counter(name for names in pages.values() for name in names)
    expected = public(package)
    if documented != expected:
        raise SystemExit(
            "the templates must document each public name once: "
            f"missing {sorted(expected - documented)}, "
            f"extra {sorted(documented - expected)}"
        )

    reference = Reference(package, pages)
    # A page whose template is gone must not survive in the output.
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    for page, source in templates.items():
        lines = []
        for line in source.splitlines():
            lines += (
                render(line[4:], 3, reference) if line.startswith("::: ") else [line]
            )
        (OUT / f"{page}.md").write_text(
            "\n".join(lines).rstrip("\n") + "\n", encoding="utf-8"
        )
    if warnings.buffer:
        raise SystemExit(
            "griffe warnings:\n" + "\n".join(r.getMessage() for r in warnings.buffer)
        )


if __name__ == "__main__":
    main()
