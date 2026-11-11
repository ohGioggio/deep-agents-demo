"""The package's layering rules, checked by reading the source with ``ast``.

Layers, lowest first: ``config``/``models`` → ``persistence``/``providers`` →
``core`` → ``features`` → ``startup`` → ``ui``. A module imports only from its own
layer or lower; a feature never imports another feature; only ``ui/`` talks to the
terminal (``print``, ``input``, ``signal``).
"""
import ast
import subprocess
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "deep_agents_demo"

LAYERS = {"config": 0, "models": 0, "persistence": 1, "providers": 1, "core": 2,
          "features": 3, "startup": 4, "ui": 5}

# Flat modules not moved into a layer yet (by file stem). Each task that moves one
# removes it; the last move empties the set. A legacy module isn't checked as an
# importer, and a moved module may still import it until it moves.
LEGACY: set[str] = set()

# Moved modules (dotted, under the package) that still print, read input or use
# signal; emptied once the UI seam is in place.
IO_PENDING: set[str] = set()

ENTRY = {"__init__", "__main__"}


def module_name(path: Path) -> str:
    """``core/feature.py`` → ``core.feature``; a package's ``__init__`` → the package."""
    parts = list(path.relative_to(PACKAGE).with_suffix("").parts)
    if parts[-1] == "__init__" and len(parts) > 1:
        parts = parts[:-1]
    return ".".join(parts)


def layer_of(module: str) -> int | None:
    """The layer of a module under the package, or None (legacy, entry, unknown)."""
    top = module.split(".")[0]
    if top in LEGACY or top in ENTRY:
        return None
    return LAYERS.get(top)


def imports_of(path: Path) -> list[str]:
    """Every ``deep_agents_demo`` module this file imports, top level or inside
    functions, relative to the package (``core.feature``)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    here = module_name(path).split(".")
    if path.name != "__init__.py":
        here = here[:-1]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here[:len(here) - node.level + 1]
                mod = ".".join(["deep_agents_demo", *base, *(node.module or "").split(".")]).rstrip(".")
            else:
                mod = node.module or ""
            if mod == "deep_agents_demo":
                found += [f"deep_agents_demo.{a.name}" for a in node.names]
            else:
                found.append(mod)
    return [m.removeprefix("deep_agents_demo.") for m in found
            if m.startswith("deep_agents_demo.")]


def modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        if "seed" in path.relative_to(PACKAGE).parts:
            continue
        yield path, module_name(path)


def feature_of(module: str) -> str | None:
    parts = module.split(".")
    return parts[1] if parts[0] == "features" and len(parts) > 1 else None


class LayerTests(unittest.TestCase):
    def test_layers_only_import_downward(self):
        wrong = []
        for path, module in modules():
            mine = layer_of(module)
            if mine is None:
                continue
            for target in imports_of(path):
                theirs = layer_of(target)
                if theirs is not None and theirs > mine:
                    wrong.append(f"{module} imports {target}")
        self.assertEqual(wrong, [])

    def test_features_do_not_import_each_other(self):
        wrong = []
        for path, module in modules():
            mine = feature_of(module)
            if mine is None:
                continue
            for target in imports_of(path):
                theirs = feature_of(target)
                if theirs is not None and theirs != mine:
                    wrong.append(f"{module} imports {target}")
        self.assertEqual(wrong, [])

    def test_terminal_io_only_in_ui(self):
        wrong = []
        for path, module in modules():
            top = module.split(".")[0]
            if layer_of(module) is None or top == "ui" or module in IO_PENDING:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Name) and node.id in ("print", "input"):
                    # a call, or the function passed on (e.g. ``say=print`` as a default)
                    wrong.append(f"{module}:{node.lineno} uses {node.id}")
                elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id == "signal"):
                    wrong.append(f"{module}:{node.lineno} uses signal.{node.attr}")
        self.assertEqual(wrong, [])

    def test_every_flat_module_is_accounted_for(self):
        flat = {p.stem for p in PACKAGE.glob("*.py")} - ENTRY - {"config"}
        self.assertEqual(flat - LEGACY, set(), "a new flat module: put it in a layer")
        self.assertEqual(LEGACY - flat, set(), "moved: take it out of LEGACY")
        self.assertEqual({m for m in IO_PENDING if not (PACKAGE / (m.replace(".", "/") + ".py")).exists()},
                         set(), "IO_PENDING names a module that doesn't exist")

    def test_entry_point(self):
        import importlib
        import tomllib
        project = tomllib.loads((PACKAGE.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
        target = project["project"]["scripts"]["deep-agents-demo"]
        self.assertEqual(target, "deep_agents_demo.ui.cli.main:main")
        module, _, name = target.partition(":")
        self.assertTrue(callable(getattr(importlib.import_module(module), name)))
        self.assertIn("from deep_agents_demo.ui.cli.main import main",
                      (PACKAGE / "__main__.py").read_text(encoding="utf-8"))
        self.assertEqual(LEGACY, set(), "every module is in a layer")

    def test_removed_modules_are_not_imported(self):
        import re
        removed = ["agent", "agent_files", "agent_manager", "agent_registry", "app", "approvals",
                   "catalog", "chat", "cli", "commands", "embeddings", "guardrails", "images", "memory",
                   "memory_policy", "openshell_backend", "orchestration", "rag", "rooms", "sandbox",
                   "skill_manager", "skill_store", "threads", "tools", "workspace"]
        names = "|".join(removed)
        # \b: "deep_agents_demo.agent" must not match "deep_agents_demo.agents"; a patch
        # target like "deep_agents_demo.images.MAX_BYTES" still matches.
        dotted = re.compile(rf"deep_agents_demo\.({names})\b")
        bare = re.compile(rf"from deep_agents_demo import ({names})\b")
        found = []
        root = PACKAGE.parents[1]
        for folder in ("src", "tests", "evals"):
            for path in (root / folder).rglob("*.py"):
                if path == Path(__file__).resolve():
                    continue  # this file names them on purpose
                text = path.read_text(encoding="utf-8")
                for pattern in (dotted, bare):
                    found += [f"{path.relative_to(root)}: {m.group(0)}" for m in pattern.finditer(text)]
        self.assertEqual(found, [])

    def test_core_imports_no_feature(self):
        # Assembling an agent once imported orchestration, which imported the chat, which
        # imported the app, which imported the agent: core must stand alone.
        code = ("import sys, deep_agents_demo.core.assembly, deep_agents_demo.core.turns; "
                "print(any(m.startswith('deep_agents_demo.features') for m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
