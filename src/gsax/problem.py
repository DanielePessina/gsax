"""Defines the Problem dataclass for sensitivity analysis."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    """Parameter names and their bounds for a sensitivity analysis problem."""

    names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    output_names: tuple[str, ...] | None = None

    @classmethod
    def from_dict(
        cls,
        params: dict[str, tuple[float, float]],
        output_names: tuple[str, ...] | None = None,
    ) -> "Problem":
        """Create a Problem from a dict mapping parameter names to bounds."""
        return cls(
            names=tuple(params.keys()),
            bounds=tuple(params.values()),
            output_names=output_names,
        )

    @property
    def num_vars(self) -> int:
        """Return the number of parameters."""
        return len(self.names)
