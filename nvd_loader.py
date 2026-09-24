"""LangChain loader for keyword searches against the NVD CVE API 2.0."""

from __future__ import annotations

import json
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class NVDLoader(BaseLoader):
    """Load a bounded set of NVD CVE records matching a keyword search.

    The public NVD API is used without an API key. ``max_cves`` is capped at
    100 to keep each request and the resulting document batch manageable.
    Network access occurs only when :meth:`lazy_load` is iterated.
    """

    API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    MAX_CVES_PER_REQUEST = 100

    def __init__(self, keyword: str, max_cves: int = 10, timeout: float = 20.0) -> None:
        """Store and validate a search term, result limit, and request timeout."""
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("keyword must be a non-empty string")
        if isinstance(max_cves, bool) or not isinstance(max_cves, int) or max_cves < 1:
            raise ValueError("max_cves must be a positive integer")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.keyword = keyword.strip()
        self.max_cves = min(max_cves, self.MAX_CVES_PER_REQUEST)
        self.timeout = timeout

    def lazy_load(self) -> Iterator[Document]:
        """Fetch the bounded keyword result set and yield one document per CVE."""
        params = urlencode(
            {"keywordSearch": self.keyword, "resultsPerPage": self.max_cves}
        )
        request = Request(
            f"{self.API_URL}?{params}",
            headers={"Accept": "application/json", "User-Agent": "Homework2-NVDLoader/1.0"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"NVD API request failed: {exc}") from exc

        for result in payload.get("vulnerabilities", [])[: self.max_cves]:
            cve = result.get("cve", {})
            cve_id = cve.get("id")
            if not cve_id:
                continue

            description = _english_description(cve.get("descriptions", []))
            products = _affected_products(cve)
            score, severity, cvss_version = _cvss_details(cve.get("metrics", {}))
            title = _title_for(cve_id, description)
            url = f"https://nvd.nist.gov/vuln/detail/{cve_id}"

            content_lines = [f"CVE ID: {cve_id}", f"Title: {title}"]
            if description:
                content_lines.append(f"Description: {description}")
            if products:
                content_lines.append(f"Affected product(s): {', '.join(products)}")
            if score is not None:
                cvss_line = f"CVSS {cvss_version or ''}: {score}"
                if severity:
                    cvss_line += f" ({severity})"
                content_lines.append(cvss_line)
            content_lines.append(f"Source: {url}")

            metadata: dict[str, Any] = {
                "source": "nvd",
                "advisory_id": cve_id,
                "title": title,
                "published": cve.get("published", ""),
                "last_modified": cve.get("lastModified", ""),
                "url": url,
            }
            if products:
                metadata["product"] = ", ".join(products)
            if severity:
                metadata["severity"] = severity
            if score is not None:
                metadata["cvss_score"] = score
            if cvss_version:
                metadata["cvss_version"] = cvss_version

            yield Document(page_content="\n".join(content_lines), metadata=metadata)


def _english_description(descriptions: list[dict[str, Any]]) -> str:
    """Return the English CVE description, falling back to the first entry."""
    for item in descriptions:
        if item.get("lang") == "en":
            return str(item.get("value", ""))
    return str(descriptions[0].get("value", "")) if descriptions else ""


def _title_for(cve_id: str, description: str) -> str:
    """Build a compact title because NVD CVE records do not define one."""
    if not description:
        return cve_id
    first_sentence = description.split(". ", maxsplit=1)[0].rstrip(".")
    return f"{cve_id}: {first_sentence[:160]}"


def _affected_products(cve: dict[str, Any]) -> list[str]:
    """Extract product names from CVE affected data and NVD CPE configurations."""
    products: set[str] = set()

    for affected in cve.get("affected", []):
        vendor = affected.get("vendor", "")
        product = affected.get("product", "")
        if product:
            products.add(f"{vendor} {product}".strip())

    def visit_nodes(nodes: list[dict[str, Any]]) -> None:
        """Walk nested NVD configuration nodes to collect CPE vendor/product."""
        for node in nodes:
            for match in node.get("cpeMatch", []):
                criteria = match.get("criteria", "")
                parts = criteria.split(":")
                if len(parts) > 4 and parts[0] == "cpe" and parts[1] in {"2.3", "2.2"}:
                    vendor, product = parts[3], parts[4]
                    if product not in {"*", "-"}:
                        products.add(f"{vendor} {product}".replace("_", " ").strip())
            visit_nodes(node.get("children", []))

    visit_nodes(cve.get("configurations", []))
    return sorted(products)


def _cvss_details(metrics: dict[str, Any]) -> tuple[float | None, str | None, str | None]:
    """Select the newest available CVSS metric and return score/severity/version."""
    for version, key in (
        ("4.0", "cvssMetricV40"),
        ("3.1", "cvssMetricV31"),
        ("3.0", "cvssMetricV30"),
        ("2.0", "cvssMetricV2"),
    ):
        entries = metrics.get(key, [])
        if entries:
            primary = next((entry for entry in entries if entry.get("type") == "Primary"), entries[0])
            data = primary.get("cvssData", {})
            score = data.get("baseScore")
            if score is None:
                continue
            severity = primary.get("baseSeverity") or data.get("baseSeverity")
            return float(score), str(severity) if severity else None, version
    return None, None, None
