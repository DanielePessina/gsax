"""Write the gsax Claude Code skill file into a target directory."""

from __future__ import annotations

import importlib.resources
import shutil
from pathlib import Path


def write_skill(dest: str | Path | None = None) -> Path:
    """Copy the gsax skill file to *dest* (defaults to cwd).

    Returns the path of the written file.
    """
    if dest is None:
        dest = Path.cwd()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    src = importlib.resources.files("gsax._skill").joinpath("gsax.md")
    out = dest / "gsax.md"
    with importlib.resources.as_file(src) as src_path:
        shutil.copy2(src_path, out)

    print(f"Wrote skill file to: {out}\n")
    print("To install the skill:\n")
    print("  Claude Code (project):  copy to .claude/commands/gsax.md")
    print("  Claude Code (global):   copy to ~/.claude/commands/gsax.md")
    print("  Codex (project):        copy to .codex/commands/gsax.md\n")
    print("On Windows, use backslashes:")
    print("  Claude Code (project):  copy to .claude\\commands\\gsax.md")
    print("  Claude Code (global):   copy to %USERPROFILE%\\.claude\\commands\\gsax.md")
    print("  Codex (project):        copy to .codex\\commands\\gsax.md")

    return out
