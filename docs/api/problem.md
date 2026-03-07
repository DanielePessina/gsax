# Problem

Immutable dataclass defining the parameter space.

## Definition

```python
@dataclass(frozen=True)
class Problem:
    names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `names` | `tuple[str, ...]` | Parameter name for each of the D dimensions. |
| `bounds` | `tuple[tuple[float, float], ...]` | Lower and upper bound for each parameter. Length must match `names`. |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_vars` | `int` | D -- the number of parameters. Equivalent to `len(names)`. |

## Class Methods

### `Problem.from_dict()`

Construct a Problem from a dictionary.

```python
@classmethod
def from_dict(cls, params: dict[str, tuple[float, float]]) -> Problem
```

Keys become `names`, values become `bounds`.

## Examples

```python
from gsax import Problem

# From a dictionary
problem = Problem.from_dict({
    "x1": (-3.14, 3.14),
    "x2": (-3.14, 3.14),
    "x3": (-3.14, 3.14),
})

# Directly
problem = Problem(
    names=("x1", "x2", "x3"),
    bounds=((-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)),
)

print(problem.num_vars)  # 3
```
