"""Small structural checks for the method registry.

Numerical capability behavior belongs in the method integration tests. This
module only protects registry completeness and declaration shape.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import jaxgsa
from jaxgsa._core.registry import MethodSpec, methods, register

METHOD_ROOT = Path(jaxgsa.__file__).parent
NOT_METHODS = {"_core", "benchmarks", "__pycache__"}


def _method_packages() -> set[str]:
    """Discover method packages from the installed source tree."""
    return {
        path.name
        for path in METHOD_ROOT.iterdir()
        if path.is_dir() and path.name not in NOT_METHODS and (path / "__init__.py").exists()
    }


def test_every_method_package_is_registered_and_exported() -> None:
    """A new method cannot disappear between disk, registry, and root API."""
    registered = set(methods())
    assert _method_packages() == registered
    assert registered <= set(jaxgsa.__all__)
    assert all(getattr(jaxgsa, name, None) is not None for name in registered)


def test_registry_declarations_use_supported_capability_values() -> None:
    """Registry metadata stays within the finite public vocabulary."""
    for name, spec in methods().items():
        assert spec.name == name
        assert spec.correlation in {"accepts", "refuses"}, name
        assert spec.categorical in {"accepts", "refuses"}, name
        assert spec.bootstrap in {None, "n_bootstrap"}, name
        assert spec.invalid_unit is None or spec.invalid_unit.value, name


def test_pure_core_declarations_match_namespace_exports() -> None:
    """Every declared pure core exists; only the two host methods are exempt."""
    exempt = {name for name, spec in methods().items() if not spec.pure_core}
    assert exempt == {"kucherenko", "vkoga"}
    for name, spec in methods().items():
        module = importlib.import_module(f"jaxgsa.{name}")
        has_core = hasattr(module, "indices") and "indices" in module.__all__
        assert has_core is spec.pure_core, name


def test_duplicate_registration_is_rejected_but_idempotent_registration_is_safe() -> None:
    """Registry imports are idempotent and conflicting names fail loudly."""
    existing = methods()["sobol"]
    assert register(existing) is existing
    duplicate = MethodSpec(
        name="sobol",
        analyze=existing.analyze,
        sample=None,
        result=existing.result,
        correlation="accepts",
        categorical="accepts",
        bootstrap=None,
        invalid_unit=None,
    )
    with pytest.raises(ValueError, match="already registered"):
        register(duplicate)
