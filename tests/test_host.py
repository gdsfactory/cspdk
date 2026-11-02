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
    for name in ("sm", "xp", "solve", "port_namer", "interpolate"):
        assert hasattr(host, name)
    assert callable(host.differentiable)
    assert callable(host.set_host)
    assert callable(host.get_host)


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
