"""Parallel SSRF-safe brand and product photo scraper.

Implements F9 & P8:
- Runs optional inspiration_url breakdown in parallel with brand_url scrape via asyncio.gather.
- Fetches homepage HTML ONCE and reuses the parsed tree for brand text, hex colors,
  og:image/twitter:image, JSON-LD Product.image, and product page link discovery.
- Concurrently probes Shopify /products.json and discovered /product|/shop|/store pages
  with bounded asyncio.Semaphore concurrency.
- Validates EVERY URL (homepage, inspiration, Shopify JSON, product links, image URLs)
  through M1's SSRF guard (assert_public_url + ResilientHttpClient public_only=True).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.ssrf import SSRFError, assert_public_url
from gtm_videogen.models.schemas import MediaRef, ResearchBundle

_HEX_COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_PRODUCT_LINK_RE = re.compile(r"/(?:products?|shop|store)(?:/|$|\?)", re.IGNORECASE)
_EXCLUDED_IMAGE_RE = re.compile(
    r"(?:\.(?:svg|ico)(?:$|\?)|(?:^|[/._-])(?:pixel|tracker|spacer|sprite|favicon|1x1)(?:[/._-]|$))",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_TEXT_CHARS = 12_000


def _normalize_hex(raw: str) -> str | None:
    cleaned = raw.strip()
    if not cleaned.startswith("#"):
        return None
    body = cleaned[1:]
    if len(body) == 3 and all(c in "0123456789abcdefABCDEF" for c in body):
        body = "".join(ch * 2 for ch in body)
    if len(body) == 6 and all(c in "0123456789abcdefABCDEF" for c in body):
        return f"#{body.upper()}"
    return None


def _clean_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _infer_mime_type(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".png"):
        return "image/png"
    if path.endswith(".webp"):
        return "image/webp"
    if path.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def _is_usable_photo_url(url: str) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return False
    return not bool(_EXCLUDED_IMAGE_RE.search(url))


def _dedupe_preserve_order(items: list[str], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        norm = _clean_text(item)
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
        if limit is not None and len(out) >= limit:
            break
    return out


def _extract_hex_colors(soup: BeautifulSoup) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def _add(raw_hex: str) -> None:
        norm = _normalize_hex(raw_hex)
        if norm and norm not in seen:
            seen.add(norm)
            found.append(norm)

    # 1. <meta name="theme-color">
    for meta in soup.find_all("meta", attrs={"name": re.compile(r"^theme-color$", re.IGNORECASE)}):
        content = meta.get("content")
        if isinstance(content, str):
            for match in _HEX_COLOR_RE.findall(content):
                _add(match)

    # 2. <style> blocks
    for style_tag in soup.find_all("style"):
        css_text = style_tag.get_text(" ", strip=True)
        for match in _HEX_COLOR_RE.findall(css_text):
            _add(match)

    # 3. Inline style attributes
    for elem in soup.find_all(attrs={"style": True}):
        style_val = elem.get("style")
        if isinstance(style_val, str):
            for match in _HEX_COLOR_RE.findall(style_val):
                _add(match)

    return found[:8]


def _extract_meta_images(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Extract (absolute_url, alt) pairs from og:image and twitter:image meta tags."""
    results: list[tuple[str, str]] = []
    alt_tag = soup.find("meta", attrs={"property": "og:image:alt"})
    default_alt = str(alt_tag.get("content") or "").strip() if alt_tag else ""

    for attr_name, key in (("property", "og:image"), ("name", "twitter:image")):
        for meta in soup.find_all("meta", attrs={attr_name: key}):
            raw_url = meta.get("content")
            if isinstance(raw_url, str) and raw_url.strip():
                abs_url = urljoin(base_url, raw_url.strip())
                results.append((abs_url, default_alt or "OpenGraph image"))
    return results


def _iter_json_ld_nodes(data: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if isinstance(data, dict):
        nodes.append(data)
        graph = data.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                nodes.extend(_iter_json_ld_nodes(item))
    elif isinstance(data, list):
        for item in data:
            nodes.extend(_iter_json_ld_nodes(item))
    return nodes


def _extract_image_urls_from_field(value: Any, base_url: str) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str) and value.strip():
        urls.append(urljoin(base_url, value.strip()))
    elif isinstance(value, dict):
        for key in ("url", "contentUrl"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                urls.append(urljoin(base_url, candidate.strip()))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_extract_image_urls_from_field(item, base_url))
    return urls


def _extract_json_ld(soup: BeautifulSoup, base_url: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Extract Product.image URLs and product facts from JSON-LD script blocks."""
    images: list[tuple[str, str]] = []
    facts: list[str] = []

    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.IGNORECASE)}):
        raw_json = script.string or script.get_text()
        if not raw_json or not raw_json.strip():
            continue
        try:
            parsed = json.loads(raw_json)
        except (json.JSONDecodeError, ValueError):
            continue

        for node in _iter_json_ld_nodes(parsed):
            raw_type = node.get("@type")
            types = (
                [t.lower() for t in raw_type if isinstance(t, str)]
                if isinstance(raw_type, list)
                else ([raw_type.lower()] if isinstance(raw_type, str) else [])
            )
            is_product = any("product" in t for t in types)
            if not is_product:
                continue

            name = _clean_text(str(node.get("name") or ""))
            desc = _clean_text(str(node.get("description") or ""))
            if name and desc:
                facts.append(f"{name}: {desc[:220]}")
            elif name:
                facts.append(name)

            offers = node.get("offers")
            if isinstance(offers, dict):
                price = offers.get("price")
                currency = offers.get("priceCurrency", "USD")
                if price is not None:
                    facts.append(f"Listed price: {price} {currency}")

            for img_url in _extract_image_urls_from_field(node.get("image"), base_url):
                images.append((img_url, name or "Product photo"))

    return images, facts


def _extract_brand_name(soup: BeautifulSoup, brand_url: str, explicit_name: str | None) -> str:
    if explicit_name and explicit_name.strip():
        return explicit_name.strip()

    og_site = soup.find("meta", attrs={"property": "og:site_name"})
    if og_site and isinstance(og_site.get("content"), str) and og_site["content"].strip():
        return og_site["content"].strip()

    if soup.title and soup.title.string and soup.title.string.strip():
        raw_title = _clean_text(soup.title.string)
        for sep in (" | ", " - ", " — ", " · "):
            if sep in raw_title:
                return raw_title.split(sep, 1)[0].strip()
        return raw_title[:120]

    host = urlparse(brand_url).hostname or "Brand"
    return host.replace("www.", "").split(".")[0].capitalize()


_BOILERPLATE_RE = re.compile(
    r"^(log\s*in|sign\s*in|sign\s*up|log\s*in\s+or\s+sign\s*up|get\s+started|menu|"
    r"skip\s+to\s+(main\s+)?content|accept(\s+all)?\s+cookies?|cookie\s+policy|"
    r"privacy\s+policy|terms\s+of\s+(service|use)|all\s+rights\s+reserved)\b",
    re.IGNORECASE,
)


def _is_boilerplate_text(text: str) -> bool:
    t = text.strip()
    if not t or _BOILERPLATE_RE.search(t):
        return True
    low = t.lower()
    return any(
        phrase in low
        for phrase in (
            "we use cookies",
            "manage cookie",
            "log in or sign up",
            "already have an account",
        )
    )


def _extract_page_facts(soup: BeautifulSoup) -> list[str]:
    facts: list[str] = []

    meta_desc = soup.find("meta", attrs={"name": re.compile(r"^description$", re.IGNORECASE)})
    if meta_desc and isinstance(meta_desc.get("content"), str):
        desc = _clean_text(meta_desc["content"])
        if len(desc) >= 12 and not _is_boilerplate_text(desc):
            facts.append(desc[:300])

    og_desc = soup.find("meta", attrs={"property": "og:description"})
    if og_desc and isinstance(og_desc.get("content"), str):
        desc = _clean_text(og_desc["content"])
        if len(desc) >= 12 and not _is_boilerplate_text(desc):
            facts.append(desc[:300])

    for tag_name in ("h1", "h2"):
        for heading in soup.find_all(tag_name):
            text = _clean_text(heading.get_text(" ", strip=True))
            if 8 <= len(text) <= 200 and not _is_boilerplate_text(text):
                facts.append(text)

    char_budget = 0
    for para in soup.find_all(["p", "li"]):
        text = _clean_text(para.get_text(" ", strip=True))
        if 20 <= len(text) <= 280 and not _is_boilerplate_text(text):
            facts.append(text)
            char_budget += len(text)
            if char_budget >= _MAX_TEXT_CHARS or len(facts) >= 24:
                break

    return _dedupe_preserve_order(facts, limit=16)


def _discover_product_links(soup: BeautifulSoup, base_url: str, *, limit: int) -> list[str]:
    base_host = (urlparse(base_url).hostname or "").lower()
    seen: set[str] = set()
    links: list[str] = []

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)
        if parsed.scheme.lower() not in ("http", "https"):
            continue
        if (parsed.hostname or "").lower() != base_host:
            continue
        if not _PRODUCT_LINK_RE.search(parsed.path):
            continue
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if normalized in seen or normalized.rstrip("/") == base_url.rstrip("/"):
            continue
        seen.add(normalized)
        links.append(normalized)
        if len(links) >= limit:
            break

    return links


@dataclass
class _SiteScrapeResult:
    brand_name: str
    product_facts: list[str] = field(default_factory=list)
    verified_claims: list[str] = field(default_factory=list)
    brand_colors: list[str] = field(default_factory=list)
    product_photos: list[MediaRef] = field(default_factory=list)


class BrandScraper:
    """Parallel, SSRF-safe brand website & product photo scraper (F9 / P8)."""

    def __init__(
        self,
        http_client: ResilientHttpClient | None = None,
        *,
        max_concurrency: int = 4,
        max_product_pages: int = 4,
        max_photos: int = 4,
        resolve_dns_for_ssrf: bool = True,
    ) -> None:
        self._http_client = http_client
        self._max_concurrency = max(1, max_concurrency)
        self._max_product_pages = max(0, max_product_pages)
        self._max_photos = max(1, max_photos)
        self._resolve_dns_for_ssrf = resolve_dns_for_ssrf

    def _validate_public(self, url: str) -> str:
        return assert_public_url(url, resolve_dns=self._resolve_dns_for_ssrf)

    async def scrape_brand(
        self,
        brand_url: str,
        inspiration_url: str | None = None,
        cancel: CancellationToken | None = None,
        brand_name: str | None = None,
    ) -> ResearchBundle:
        """Scrape brand site (and optional inspiration URL in parallel) into a ResearchBundle."""
        if cancel is not None:
            cancel.raise_if_cancelled()

        validated_brand_url = self._validate_public(brand_url)
        validated_insp_url: str | None = None
        if inspiration_url and inspiration_url.strip():
            validated_insp_url = self._validate_public(inspiration_url)

        client = self._http_client
        owns_client = client is None
        if client is None:
            client = ResilientHttpClient(resolve_dns_for_ssrf=self._resolve_dns_for_ssrf)

        try:
            if validated_insp_url is not None:
                site_res, insp_notes = await asyncio.gather(
                    self._scrape_brand_site(
                        client,
                        validated_brand_url,
                        brand_name=brand_name,
                        cancel=cancel,
                    ),
                    self._scrape_inspiration(client, validated_insp_url, cancel=cancel),
                )
            else:
                site_res = await self._scrape_brand_site(
                    client,
                    validated_brand_url,
                    brand_name=brand_name,
                    cancel=cancel,
                )
                insp_notes = []
        finally:
            if owns_client:
                await client.aclose()

        return ResearchBundle(
            brand_url=validated_brand_url,
            brand_name=site_res.brand_name,
            verified_claims=site_res.verified_claims,
            product_facts=site_res.product_facts,
            brand_colors=site_res.brand_colors,
            product_photos=site_res.product_photos,
            inspiration_notes=insp_notes,
        )

    async def _scrape_brand_site(
        self,
        client: ResilientHttpClient,
        brand_url: str,
        *,
        brand_name: str | None,
        cancel: CancellationToken | None,
    ) -> _SiteScrapeResult:
        if cancel is not None:
            cancel.raise_if_cancelled()

        # Fetch homepage HTML ONCE and reuse parsed tree for all homepage extractions (P8).
        homepage_html = await client.get_text(brand_url, public_only=True, cancel=cancel)
        soup = BeautifulSoup(homepage_html, "lxml")

        resolved_name = _extract_brand_name(soup, brand_url, brand_name)
        brand_colors = _extract_hex_colors(soup)
        homepage_facts = _extract_page_facts(soup)
        meta_images = _extract_meta_images(soup, brand_url)
        ld_images, ld_facts = _extract_json_ld(soup, brand_url)
        product_links = _discover_product_links(soup, brand_url, limit=self._max_product_pages)

        # Concurrently probe Shopify /products.json and discovered product pages (P8).
        parsed = urlparse(brand_url)
        shopify_url = f"{parsed.scheme}://{parsed.netloc}/products.json"
        sem = asyncio.Semaphore(self._max_concurrency)

        tasks: list[asyncio.Task[tuple[list[tuple[str, str, str]], list[str]]]] = []
        tasks.append(
            asyncio.create_task(self._probe_shopify_products(client, shopify_url, sem, cancel))
        )
        for link in product_links:
            tasks.append(
                asyncio.create_task(self._probe_product_page(client, link, sem, cancel))
            )

        extra_results = await asyncio.gather(*tasks, return_exceptions=True)

        candidate_photos: list[tuple[str, str, str]] = []
        for url, alt in meta_images:
            candidate_photos.append((url, alt, "og:image"))
        for url, alt in ld_images:
            candidate_photos.append((url, alt, "json-ld"))

        all_facts = list(homepage_facts) + list(ld_facts)
        for res in extra_results:
            if isinstance(res, asyncio.CancelledError):
                raise res
            if isinstance(res, BaseException):
                continue
            page_photos, page_facts = res
            candidate_photos.extend(page_photos)
            all_facts.extend(page_facts)

        deduped_facts = _dedupe_preserve_order(all_facts, limit=24)
        verified_claims = deduped_facts[:8]
        photos = self._finalize_photos(candidate_photos)

        return _SiteScrapeResult(
            brand_name=resolved_name,
            product_facts=deduped_facts,
            verified_claims=verified_claims,
            brand_colors=brand_colors,
            product_photos=photos,
        )

    async def _probe_shopify_products(
        self,
        client: ResilientHttpClient,
        shopify_url: str,
        sem: asyncio.Semaphore,
        cancel: CancellationToken | None,
    ) -> tuple[list[tuple[str, str, str]], list[str]]:
        if cancel is not None:
            cancel.raise_if_cancelled()
        try:
            validated = self._validate_public(shopify_url)
        except SSRFError:
            return [], []

        async with sem:
            if cancel is not None:
                cancel.raise_if_cancelled()
            try:
                data = await client.get_json(validated, public_only=True, cancel=cancel)
            except asyncio.CancelledError:
                raise
            except Exception:
                return [], []

        if not isinstance(data, dict) or not isinstance(data.get("products"), list):
            return [], []

        photos: list[tuple[str, str, str]] = []
        facts: list[str] = []
        for product in data["products"][:8]:
            if not isinstance(product, dict):
                continue
            title = _clean_text(str(product.get("title") or ""))
            if title:
                facts.append(title)
            body_html = product.get("body_html")
            if isinstance(body_html, str) and body_html.strip():
                body_text = _clean_text(BeautifulSoup(body_html, "lxml").get_text(" ", strip=True))
                if body_text:
                    facts.append(body_text[:220])
            images = product.get("images")
            if isinstance(images, list):
                for img in images[:2]:
                    if isinstance(img, dict) and isinstance(img.get("src"), str):
                        photos.append(
                            (
                                urljoin(validated, img["src"]),
                                title or "Shopify product photo",
                                "shopify_products_json",
                            )
                        )
        return photos, facts

    async def _probe_product_page(
        self,
        client: ResilientHttpClient,
        page_url: str,
        sem: asyncio.Semaphore,
        cancel: CancellationToken | None,
    ) -> tuple[list[tuple[str, str, str]], list[str]]:
        if cancel is not None:
            cancel.raise_if_cancelled()
        try:
            validated = self._validate_public(page_url)
        except SSRFError:
            return [], []

        async with sem:
            if cancel is not None:
                cancel.raise_if_cancelled()
            try:
                html = await client.get_text(validated, public_only=True, cancel=cancel)
            except asyncio.CancelledError:
                raise
            except Exception:
                return [], []

        soup = BeautifulSoup(html, "lxml")
        photos: list[tuple[str, str, str]] = []
        for url, alt in _extract_meta_images(soup, validated):
            photos.append((url, alt, "product_page_og"))
        ld_imgs, ld_facts = _extract_json_ld(soup, validated)
        for url, alt in ld_imgs:
            photos.append((url, alt, "product_page_json_ld"))
        for img in soup.find_all("img", src=True)[:6]:
            src = str(img["src"]).strip()
            if src:
                abs_url = urljoin(validated, src)
                alt = _clean_text(str(img.get("alt") or "Product image"))
                photos.append((abs_url, alt, "product_page_img"))

        facts = ld_facts + _extract_page_facts(soup)[:4]
        return photos, facts

    async def _scrape_inspiration(
        self,
        client: ResilientHttpClient,
        inspiration_url: str,
        cancel: CancellationToken | None,
    ) -> list[str]:
        if cancel is not None:
            cancel.raise_if_cancelled()
        validated = self._validate_public(inspiration_url)
        try:
            html = await client.get_text(validated, public_only=True, cancel=cancel)
        except asyncio.CancelledError:
            raise
        except Exception:
            return [f"Inspiration reference URL: {validated}"]

        soup = BeautifulSoup(html, "lxml")
        notes: list[str] = []
        if soup.title and soup.title.string:
            title = _clean_text(soup.title.string)
            if title:
                notes.append(f"Inspiration title: {title}")

        for fact in _extract_page_facts(soup)[:6]:
            notes.append(fact)

        return _dedupe_preserve_order(notes, limit=8)

    def _finalize_photos(self, candidates: list[tuple[str, str, str]]) -> list[MediaRef]:
        seen: set[str] = set()
        refs: list[MediaRef] = []

        for raw_url, alt, source in candidates:
            if not _is_usable_photo_url(raw_url):
                continue
            try:
                safe_url = self._validate_public(raw_url)
            except SSRFError:
                continue
            if safe_url in seen:
                continue
            seen.add(safe_url)
            refs.append(
                MediaRef(
                    id=f"photo_{len(refs) + 1}",
                    url=safe_url,
                    mime_type=_infer_mime_type(safe_url),
                    alt=alt[:200],
                    source=source,
                )
            )
            if len(refs) >= self._max_photos:
                break

        return refs
