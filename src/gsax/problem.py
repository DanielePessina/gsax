"""Defines the Problem dataclass for sensitivity analysis."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    """Parameter names and their bounds for a sensitivity analysis problem."""

    names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]

    @classmethod
    def from_dict(cls, params: dict[str, tuple[float, float]]) -> "Problem":
        """Create a Problem from a dict mapping parameter names to bounds."""
        return cls(names=tuple(params.keys()), bounds=tuple(params.values()))

    @property
    def num_vars(self) -> int:
        """Return the number of parameters."""
        return len(self.names)
