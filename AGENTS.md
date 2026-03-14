# INSTRUCTIONS

Whenever corrected, after making a mistake or misinterpreting, add a section in here ( AGENTS.md ) to instruct future sessions, avoiding the mistake again.

ALWAYS use subagents where possible, parralel work is better.

## Tools

ALWAYS use uv in python, uv add for installs, NEVER pip install directly. uv run example.py for running, NEVER python example.py or pythons example.py directly.
Avoid editing the pyproject.toml directly, where possible use uv add , uv remove etc.
Use ruff for formatting python files, run via uv run ruff . Run ruff check on any new files before running them or including them. Fix any warnings or errors before proceeding.
Run type checking with ty on all new code. run via uv run ty only on the edited file(s). Always run after any changes to python code, fix the errors before proceeding, avoid suppressing types where possible (e.g. # noqa, # type: ignore )

## Guidelines

• Use google docstring format
• Follow SOLID design priciniples where possible
• we do not care about test coverage too much, functionality is most important.
• if writing unit tests, always use pytest , run via uv run pytest

## Session Corrections

• If the user says to use an open `tmux` session, inject commands into that live session or pane directly rather than creating a detached window or separate exec session.
• `ty` must be invoked with a subcommand. Use `uv run ty check <path>` for package or file typechecking, not `uv run ty <path>`.
• If the user says to “ignore” a typing issue in this context, confirm whether they mean “leave it unresolved” or “fix it by adding the proper guard/assertion.” Do not assume they want the diagnostic left in place.
• For `gsax` input-distribution extensions, prefer `TypedDict` parameter specs in `Problem.from_dict(...)` over runtime input-class hierarchies. Gaussian specs may include optional `low` and/or `high` truncation bounds, while `mean` and `variance` still describe the parent Gaussian before truncation.
