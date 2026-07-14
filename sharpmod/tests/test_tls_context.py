"""Tests for merging public, platform, and organization certificate roots."""

from __future__ import annotations

from sharpmod.io import tls


class _Context:
    def __init__(self):
        self.loaded = []

    def load_verify_locations(self, *, cafile):
        self.loaded.append(cafile)


def test_context_starts_with_system_roots_and_adds_certifi(monkeypatch):
    context = _Context()
    calls = []

    def default_context(*args, **kwargs):
        calls.append((args, kwargs))
        return context

    monkeypatch.setattr(tls.ssl, "create_default_context", default_context)
    monkeypatch.setattr(tls, "_CERTIFI_CA_FILE", "public-roots.pem")
    monkeypatch.delenv("SHARPMOD_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    assert tls.create_ssl_context() is context
    assert calls == [((), {})]
    assert context.loaded == ["public-roots.pem"]


def test_context_adds_application_specific_ca_bundle(monkeypatch):
    context = _Context()
    monkeypatch.setattr(
        tls.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(tls, "_CERTIFI_CA_FILE", "public-roots.pem")
    monkeypatch.setenv("SHARPMOD_CA_BUNDLE", "organization-roots.pem")
    monkeypatch.setenv("SSL_CERT_FILE", "ignored-lower-priority.pem")

    assert tls.create_ssl_context() is context
    assert context.loaded == ["public-roots.pem", "organization-roots.pem"]
