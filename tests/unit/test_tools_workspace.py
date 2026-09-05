"""The workspace guard.

Path handling is where filesystem tools actually go wrong, so these tests are
adversarial rather than illustrative: each one is a way out of the workspace
that has worked against real software.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aura.tools.workspace import PathNotAllowedError, Workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "var" / "secrets").mkdir(parents=True)
    (root / "src" / "main.py").write_text("print('hi')")
    (root / "var" / "secrets" / "api_token").write_text("SEKRIT")
    (root / ".env").write_text("KEY=1")
    (tmp_path / "outside.txt").write_text("not yours")
    # A directory whose name merely starts with the root's name.
    (tmp_path / "project-evil").mkdir()
    (tmp_path / "project-evil" / "loot.txt").write_text("x")
    return Workspace.from_config([str(root)])


def test_allows_a_file_inside_the_workspace(ws: Workspace, tmp_path: Path) -> None:
    assert ws.resolve(tmp_path / "project" / "src" / "main.py").name == "main.py"


def test_allows_a_path_that_does_not_exist_yet(ws: Workspace, tmp_path: Path) -> None:
    """Write targets have to resolve before they exist."""
    assert ws.resolve(tmp_path / "project" / "src" / "new.py").name == "new.py"


def test_must_exist_is_enforced_when_asked(ws: Workspace, tmp_path: Path) -> None:
    with pytest.raises(PathNotAllowedError, match="does not exist"):
        ws.resolve(tmp_path / "project" / "nope.txt", must_exist=True)


# -- escaping ---------------------------------------------------------------


def test_rejects_dot_dot_traversal(ws: Workspace, tmp_path: Path) -> None:
    """The classic hole: the string looks contained, the target is not."""
    escape = tmp_path / "project" / "src" / ".." / ".." / "outside.txt"
    with pytest.raises(PathNotAllowedError, match="outside the workspace"):
        ws.resolve(escape)


def test_rejects_an_absolute_path_elsewhere(ws: Workspace, tmp_path: Path) -> None:
    with pytest.raises(PathNotAllowedError, match="outside the workspace"):
        ws.resolve(tmp_path / "outside.txt")


def test_rejects_a_sibling_directory_sharing_the_root_prefix(ws: Workspace, tmp_path: Path) -> None:
    """`/x/project-evil` must not pass a containment check against `/x/project`.

    A plain startswith() without a separator lets this through.
    """
    with pytest.raises(PathNotAllowedError, match="outside the workspace"):
        ws.resolve(tmp_path / "project-evil" / "loot.txt")


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
def test_rejects_a_symlink_pointing_out_of_the_workspace(ws: Workspace, tmp_path: Path) -> None:
    """Containment is checked against the resolved target, not the link."""
    link = tmp_path / "project" / "escape.txt"
    link.symlink_to(tmp_path / "outside.txt")
    with pytest.raises(PathNotAllowedError, match="outside the workspace"):
        ws.resolve(link)


def test_no_roots_means_nothing_is_accessible(tmp_path: Path) -> None:
    """An unconfigured AURA can touch nothing, rather than everything."""
    empty = Workspace.from_config([])
    with pytest.raises(PathNotAllowedError, match="No workspace roots"):
        empty.resolve(tmp_path / "anything.txt")


# -- secrets ----------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "var/secrets/api_token",
        ".env",
    ],
)
def test_refuses_secrets_inside_the_workspace(ws: Workspace, tmp_path: Path, relative: str) -> None:
    """AURA's own tokens live in the workspace; a read tool must not reach them."""
    with pytest.raises(PathNotAllowedError, match="deny pattern"):
        ws.resolve(tmp_path / "project" / relative)


@pytest.mark.parametrize("name", ["server.pem", "id_rsa", "id_ed25519", "deploy.key"])
def test_refuses_key_material_anywhere_in_the_workspace(
    ws: Workspace, tmp_path: Path, name: str
) -> None:
    target = tmp_path / "project" / "src" / name
    target.write_text("-----BEGIN PRIVATE KEY-----")
    with pytest.raises(PathNotAllowedError, match="deny pattern"):
        ws.resolve(target)


def test_a_normal_file_named_like_a_secret_prefix_is_fine(ws: Workspace, tmp_path: Path) -> None:
    """`environment.py` must not be caught by the `.env` pattern."""
    target = tmp_path / "project" / "src" / "environment.py"
    target.write_text("x")
    assert ws.resolve(target).name == "environment.py"


# -- platform ---------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="case-insensitivity is a Windows concern")
def test_case_differences_do_not_escape_on_windows(ws: Workspace, tmp_path: Path) -> None:
    """`C:\\Users` and `c:\\users` are one directory; containment must agree."""
    weird = Path(str(tmp_path / "project" / "src" / "main.py").upper())
    assert ws.resolve(weird).exists()


def test_describe_reports_the_configuration(ws: Workspace) -> None:
    described = ws.describe()
    assert described["roots"]
    assert any(".env" in p for p in described["deny_patterns"])
