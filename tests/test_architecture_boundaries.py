"""Executable Clean Architecture dependency rules."""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("layer", "forbidden_roots"),
    [
        ("entities", {"adapters", "interfaces", "mujoco", "use_cases"}),
        ("interfaces", {"adapters", "mujoco", "use_cases"}),
        ("use_cases", {"adapters", "mujoco"}),
    ],
)
def test_dependencies_point_inward(layer, forbidden_roots):
    violations = []
    for source_path in sorted((PROJECT_ROOT / layer).rglob("*.py")):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            imported_roots = []
            if isinstance(node, ast.Import):
                imported_roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots = [node.module.split(".")[0]]
            for imported_root in imported_roots:
                if imported_root in forbidden_roots:
                    violations.append(
                        f"{source_path.relative_to(PROJECT_ROOT)} imports "
                        f"{imported_root}"
                    )

    assert not violations, "\n".join(violations)
