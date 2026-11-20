"""SSRF protection helpers for validating public HTTP/HTTPS URLs and redirect chains."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Sequence
from urllib.parse import urljoin, urlparse

ALLOWED_SCHEMES = frozenset({"http", "https"})
ALLOWED_PORTS = frozenset({80, 443})
BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")
MAX_REDIRECT_HOPS = 3


class SSRFError(ValueError):
    """Raised when a URL fails SSRF safety validation."""


def is_public_ip(ip_str: str) -> bool:
    """Return True only if ip_str is a globally routable public IPv4 or IPv6 address."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped

    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    ):
        return False

    # Carrier-grade NAT (100.64.0.0/10) is non-public even if is_global varies by Python version
    if isinstance(addr, ipaddress.IPv4Address) and addr in ipaddress.ip_network("100.64.0.0/10"):
        return False

    return addr.is_global


_UNICODE_DOT_TRANSLATION = str.maketrans({"\u3002": ".", "\uff0e": ".", "\uff61": "."})


def _normalize_hostname(raw_hostname: str) -> str:
    """Reject control/whitespace characters and normalize Unicode dot-homographs via IDNA."""
    if not raw_hostname:
        raise SSRFError("URL is missing a hostname")

    if any(
        ch.isspace() or not ch.isprintable() or ord(ch) < 0x20 or ord(ch) == 0x7F
        for ch in raw_hostname
    ):
        raise SSRFError(
            f"Hostname contains invalid control or whitespace characters: {raw_hostname!r}"
        )

    normalized = raw_hostname.translate(_UNICODE_DOT_TRANSLATION).lower()
    try:
        ascii_host = normalized.encode("idna").decode("ascii")
    except (UnicodeError, ValueError) as exc:
        raise SSRFError(f"Invalid IDNA hostname: {raw_hostname!r}") from exc

    ascii_host = ascii_host.lower().rstrip(".")
    if not ascii_host:
        raise SSRFError("URL is missing a hostname")
    return ascii_host


def _check_literal_ip(hostname: str) -> bool:
    """If hostname is an IP literal (canonical or alternate IPv4), validate it and return True."""
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        # Catch alternate IPv4 encodings (e.g. 127.1, 2130706433, 0x7f000001, 0177.0.0.1)
        try:
            packed = socket.inet_aton(hostname)
        except (OSError, ValueError):
            return False
        canonical_ipv4 = str(ipaddress.IPv4Address(packed))
        if not is_public_ip(canonical_ipv4):
            raise SSRFError(
                f"Blocked non-public IP address: {hostname} ({canonical_ipv4})"
            ) from None
        return True

    if not is_public_ip(hostname):
        raise SSRFError(f"Blocked non-public IP address: {hostname}")
    return True


def assert_public_url(url: str, *, resolve_dns: bool = True) -> str:
    """Validate that a URL is a public http(s) URL on port 80/443 with public IP resolution."""
    if not isinstance(url, str) or not url.strip():
        raise SSRFError("URL must be a non-empty string")

    cleaned = url.strip()
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cleaned):
        raise SSRFError("URL contains invalid control characters")

    try:
        parsed = urlparse(cleaned)
        raw_hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise SSRFError(f"Malformed URL: {cleaned!r}") from exc

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise SSRFError(f"Unsupported URL scheme: {parsed.scheme!r} (only http/https allowed)")

    if parsed.username or parsed.password:
        raise SSRFError("URLs with embedded credentials are not allowed")

    hostname = _normalize_hostname(raw_hostname or "")

    if port is not None and port not in ALLOWED_PORTS:
        raise SSRFError(f"Blocked non-standard port: {port} (only 80 and 443 allowed)")

    if hostname in BLOCKED_HOSTNAMES or hostname.endswith(BLOCKED_SUFFIXES):
        raise SSRFError(f"Blocked internal hostname: {hostname}")

    if _check_literal_ip(hostname):
        return cleaned

    if resolve_dns:
        try:
            addr_infos = socket.getaddrinfo(
                hostname, port or (443 if parsed.scheme.lower() == "https" else 80)
            )
        except OSError as exc:
            raise SSRFError(f"DNS resolution failed for {hostname}: {exc}") from exc

        if not addr_infos:
            raise SSRFError(f"DNS resolution returned no addresses for {hostname}")

        for info in addr_infos:
            resolved_ip = str(info[4][0])
            if not is_public_ip(resolved_ip):
                raise SSRFError(f"Hostname {hostname} resolved to non-public IP {resolved_ip}")

    return cleaned


def assert_public_redirect(
    current_url: str,
    location: str,
    *,
    hop: int,
    max_hops: int = MAX_REDIRECT_HOPS,
    resolve_dns: bool = True,
) -> str:
    """Validate a single redirect hop against max_hops and SSRF rules."""
    if hop > max_hops:
        raise SSRFError(f"Exceeded maximum redirect hops ({max_hops})")
    try:
        next_url = urljoin(current_url, location)
    except ValueError as exc:
        raise SSRFError(f"Malformed redirect URL: {location!r}") from exc
    return assert_public_url(next_url, resolve_dns=resolve_dns)


def validate_redirect_chain(
    urls: Sequence[str],
    *,
    max_hops: int = MAX_REDIRECT_HOPS,
    resolve_dns: bool = True,
) -> list[str]:
    """Validate an entire redirect chain (initial URL + up to max_hops redirects)."""
    if not urls:
        raise SSRFError("Redirect chain cannot be empty")
    if len(urls) - 1 > max_hops:
        raise SSRFError(f"Redirect chain has {len(urls) - 1} hops, exceeding limit of {max_hops}")

    validated: list[str] = [assert_public_url(urls[0], resolve_dns=resolve_dns)]
    for idx in range(1, len(urls)):
        validated.append(
            assert_public_redirect(
                validated[-1],
                urls[idx],
                hop=idx,
                max_hops=max_hops,
                resolve_dns=resolve_dns,
            )
        )
    return validated
