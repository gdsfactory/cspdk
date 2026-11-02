"""The model-host selector resolves the current engine lazily and forwards to it.

These tests use a stand-in host object so they do not depend on any engine being
installed: the selector's own behaviour (lazy import, precedence, scoping, forwarding)
is what is pinned here.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import cspdk._host as host


class _FakeHost:
    """A minimal stand-in carrying the protocol attributes."""

    sm = SimpleNamespace(straight="STRAIGHT")
    xp = SimpleNamespace(sin="SIN")
    differentiable = True

    @staticmethod
    def solve(connections, ports, instances):  # noqa: ANN001, ANN205
        """Return the arguments it was given, for the forwarding test."""
        return (connections, ports, instances)


def test_importing_the_selector_does_not_import_engines() -> None:
    """Importing the selector must not pull in jax or sax (checked in a fresh process)."""
    code = (
        "import sys; import cspdk._host; "
        "assert 'jax' not in sys.modules, 'jax imported'; "
        "assert 'sax' not in sys.modules, 'sax imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_protocol_members_exist() -> None:
    """Every protocol member is exposed, and the controls are callable."""
    for name in (
        "sm",
        "xp",
        "solve",
        "port_namer",
        "interpolate",
        "jit",
        "ensure_compile_time_eval",
        "reciprocal",
        "set_port_naming_strategy",
        "SDict",
        "SCoo",
        "SType",
        "FloatArrayLike",
        "FloatArray",
        "ArrayLike",
    ):
        assert hasattr(host, name)
    assert callable(host.differentiable)
    assert callable(host.set_host)
    assert callable(host.get_host)


def test_no_model_module_imports_sax_or_jax_at_module_level() -> None:
    """The model layer is fully routed: no ``import sax`` / ``import jax`` anywhere.

    The motivating invariant: a cspdk model module must be importable on an engine that
    is not sax, so it may never take ``sax`` or ``jax`` at module level. This scans the
    source (samples/demo scripts are excluded -- they are standalone entry points that
    choose their own engine, not part of the reusable model layer).
    """
    import pathlib
    import re

    import cspdk

    root = pathlib.Path(cspdk.__file__).resolve().parent
    offenders: list[str] = []
    pattern = re.compile(r"^\s*(?:import\s+(?:sax|jax)\b|from\s+(?:sax|jax)\b)")
    for path in sorted(root.rglob("*.py")):
        if "samples" in path.parts or path.name == "_host.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.match(line):
                offenders.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")
    assert not offenders, "model modules still import sax/jax:\n" + "\n".join(offenders)


def test_set_host_switches_the_resolved_host() -> None:
    """An explicit host wins and is returned by get_host."""
    fake = _FakeHost()
    previous = host._explicit
    try:
        host.set_host(fake)
        assert host.get_host() is fake
        assert host.differentiable() is True
    finally:
        host.set_host(previous)


def test_no_engine_name_registry_and_dotted_default() -> None:
    """cspdk keeps no engine-name registry; hosts are objects or dotted module paths."""
    assert not hasattr(host, "_HOSTS")
    assert host._DEFAULT == "sax.host"


def test_dotted_module_path_is_imported() -> None:
    """A dotted module path resolves to the imported module (object form also works)."""
    import sys
    import types

    module_name = "cspdk_test_fake_host_module"
    fake = types.ModuleType(module_name)
    setattr(fake, "differentiable", False)  # noqa: B010
    sys.modules[module_name] = fake
    previous = host._explicit
    try:
        host.set_host(module_name)
        assert host.get_host() is fake
        assert host.differentiable() is False
    finally:
        host.set_host(previous)
        del sys.modules[module_name]


def test_proxies_forward_attribute_access_and_calls() -> None:
    """A proxy forwards attribute access (and calls) to the resolved host."""
    fake = _FakeHost()
    previous = host._explicit
    try:
        host.set_host(fake)
        assert host.sm.straight == "STRAIGHT"
        assert host.xp.sin == "SIN"
        assert host.solve(1, 2, 3) == (1, 2, 3)
    finally:
        host.set_host(previous)


def test_host_scope_temporarily_overrides_and_reverts() -> None:
    """A host_scope overrides the env default only inside the block."""
    inside = _FakeHost()
    outside = _FakeHost()
    previous_explicit = host._explicit
    previous_env = host._ENV_INITIAL
    try:
        host.set_host(None)  # no explicit override, so the scope is consulted
        host._ENV_INITIAL = outside
        with host.host_scope(inside):
            assert host.get_host() is inside
        assert host.get_host() is outside
    finally:
        host.set_host(previous_explicit)
        host._ENV_INITIAL = previous_env
