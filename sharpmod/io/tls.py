"""TLS context helpers for user-facing HTTPS requests."""

from __future__ import annotations

import os
import ssl

try:  # certifi is a declared runtime dependency; remain import-safe anyway.
    import certifi

    _CERTIFI_CA_FILE = certifi.where()
except Exception:  # pragma: no cover - certifi is present in supported installs
    _CERTIFI_CA_FILE = None


def create_ssl_context() -> ssl.SSLContext:
    """Return a verified context containing system, public, and custom roots.

    Starting with the platform defaults preserves organization certificates
    installed on the machine.  Certifi's public roots are added rather than
    replacing that trust store.  ``SHARPMOD_CA_BUNDLE`` provides an
    application-specific override; common Python/Requests variables are also
    accepted for managed environments.
    """
    context = ssl.create_default_context()
    if _CERTIFI_CA_FILE:
        context.load_verify_locations(cafile=_CERTIFI_CA_FILE)

    custom_ca = (
        os.environ.get("SHARPMOD_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
    )
    if custom_ca and os.path.abspath(custom_ca) != os.path.abspath(
            _CERTIFI_CA_FILE or ""):
        context.load_verify_locations(cafile=custom_ca)
    return context
