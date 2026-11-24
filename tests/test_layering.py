"""Source placement rules (docs/maintainer/engine-architecture.md section 8, AGENTS.md).

Family-specific code lives in src/models/<family>/ and tools/convert/<family>*.py; shared code
never includes, names or branches on a family. src/models/registry.* and src/models/shapes.cpp are
the only shared files that name families. Ops identifiers name no model or speculative backend,
and model code, tests and benchmarks reach an Op only through its public contract.

Comments and string literals are stripped before identifiers are matched: help text and messages
may name the models a flag applies to.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP_SUFFIXES = {".h", ".hpp", ".cpp", ".cu", ".cuh", ".inl"}
FAMILIES = ("qwen3_5", "qwen4_exp", "glm5_next")
SHARED_DIRS = ("src/runtime", "src/serve", "src/product", "src/core", "src/artifact", "src/text",
               "src/media", "src/ops", "include", "apps")
FAMILY_NAMES = re.compile(r"qwen|flash_?next|27b|35b|glm", re.IGNORECASE)
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)

# The registry is the one models header the Engine reaches a family through.
SHARED_MODEL_INCLUDES = {"models/registry.h"}


def sources(*dirs: str) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        out.extend(p for p in (ROOT / d).rglob("*") if p.suffix in CPP_SUFFIXES)
    return sorted(out)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def strip_cpp(text: str) -> str:
    """Removes comments, string, character and hex literals (raw strings included)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c == "R" and text.startswith('R"', i) and (i == 0 or not text[i - 1].isalnum()):
            open_paren = text.find("(", i + 2)
            delimiter = text[i + 2:open_paren]
            close = text.find(")" + delimiter + '"', open_paren)
            i = n if close < 0 else close + len(delimiter) + 2
            out.append('""')
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            out.append('""')
        else:
            out.append(c)
            i += 1
    # Hex literals such as SHA-256's round constants can spell "27b".
    return re.sub(r"\b0[xX][0-9a-fA-F']+", "0", "".join(out))


def includes(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Includes inside comments do not count.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return INCLUDE.findall(text)


def family_of(path: Path) -> str | None:
    parts = path.relative_to(ROOT).parts
    if len(parts) > 2 and parts[0] == "src" and parts[1] == "models" and parts[2] in FAMILIES:
        return parts[2]
    return None


def test_shared_code_includes_no_model_code():
    bad = [f"{rel(p)}: {inc}" for p in sources(*SHARED_DIRS) for inc in includes(p)
           if inc.startswith("models/") and inc not in SHARED_MODEL_INCLUDES]
    assert not bad, "shared code includes model code:\n" + "\n".join(bad)


def test_lower_layers_include_no_runtime_or_serving():
    bad = [f"{rel(p)}: {inc}" for p in sources("src/ops", "src/core", "src/artifact", "include")
           for inc in includes(p)
           if inc.startswith(("runtime/", "serve/", "product/"))]
    assert not bad, "a lower layer includes runtime, serving or product code:\n" + "\n".join(bad)


def test_families_are_independent():
    bad = []
    for p in sources("src/models"):
        own = family_of(p)
        for inc in includes(p):
            for family in FAMILIES:
                if inc.startswith(f"models/{family}/") and family != own:
                    if rel(p).startswith(("src/models/registry", "src/models/shapes.cpp")):
                        continue
                    bad.append(f"{rel(p)}: {inc}")
    assert not bad, "a family (or shared model code) includes another family:\n" + "\n".join(bad)


def test_shared_model_code_is_model_independent():
    bad = []
    for p in sources("src/models/common"):
        bad += [f"{rel(p)}: {inc}" for inc in includes(p)
                if inc.startswith("models/") and not inc.startswith("models/common/")]
        code = strip_cpp(p.read_text(encoding="utf-8", errors="replace"))
        bad += [f"{rel(p)}:{n}: {line.strip()}" for n, line in enumerate(code.splitlines(), 1)
                if FAMILY_NAMES.search(line)]
    assert not bad, "src/models/common depends on or names a family:\n" + "\n".join(bad)


def test_model_code_uses_public_op_contracts():
    private = ("ops/common/", "ops/kernel/", "ops/launcher/")
    bad = []
    for p in sources("src/models"):
        if "/shapes/" in rel(p) or rel(p) == "src/models/shapes.cpp":
            continue
        for inc in includes(p):
            if inc.startswith(private) or inc.endswith("_launch.cuh"):
                bad.append(f"{rel(p)}: {inc}")
    assert not bad, "model code includes private Op headers:\n" + "\n".join(bad)


def test_tests_and_benchmarks_use_public_op_contracts():
    bad = [f"{rel(p)}: {inc}" for p in sources("tests", "bench") for inc in includes(p)
           if inc.startswith(("ops/kernel/", "ops/launcher/"))]
    assert not bad, "a test or benchmark includes a private Op header:\n" + "\n".join(bad)


def test_shared_code_names_no_family():
    bad = []
    for p in sources(*SHARED_DIRS):
        code = strip_cpp(p.read_text(encoding="utf-8", errors="replace"))
        for lineno, line in enumerate(code.splitlines(), 1):
            if FAMILY_NAMES.search(line):
                bad.append(f"{rel(p)}:{lineno}: {line.strip()}")
    assert not bad, "shared code names a model family:\n" + "\n".join(bad)


def test_ops_name_no_speculative_backend():
    bad = []
    for p in sources("src/ops", "include/infernix/ops"):
        code = strip_cpp(p.read_text(encoding="utf-8", errors="replace"))
        for lineno, line in enumerate(code.splitlines(), 1):
            if re.search(r"dflash", line, re.IGNORECASE):
                bad.append(f"{rel(p)}:{lineno}: {line.strip()}")
    assert not bad, "Ops code names a speculative backend:\n" + "\n".join(bad)


def python_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.rsplit(".", 1)[-1])
            names.update(alias.name for alias in node.names)
    return names


def test_converter_families_are_independent():
    convert = ROOT / "tools" / "convert"
    modules = {p.stem: p for p in convert.glob("*.py")}
    family_modules = {stem: next(f for f in FAMILIES if stem.startswith(f))
                      for stem in modules if stem.startswith(FAMILIES)}
    bad = []
    for stem, family in family_modules.items():
        for name in python_imports(modules[stem]):
            other = family_modules.get(name)
            if other is not None and other != family:
                bad.append(f"tools/convert/{stem}.py imports {name}")
    assert not bad, "a converter family imports another family:\n" + "\n".join(bad)
