from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]

    @classmethod
    def from_dict(cls, params: dict[str, tuple[float, float]]) -> "Problem":
        return cls(names=tuple(params.keys()), bounds=tuple(params.values()))

    @property
    def num_vars(self) -> int:
        return len(self.names)
