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
LEGACY = {
    "agent", "agent_files", "agent_manager", "agent_registry", "app", "approvals",
    "catalog", "chat", "cli", "commands", "guardrails", "images", "memory",
    "openshell_backend", "orchestration", "rag", "rooms", "sandbox",
    "skill_manager", "workspace",
}

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
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in ("print", "input")):
                    wrong.append(f"{module}:{node.lineno} calls {node.func.id}()")
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

    def test_no_import_cycle_through_agent(self):
        code = ("import sys, deep_agents_demo.core.feature; "
                "print('deep_agents_demo.orchestration' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
