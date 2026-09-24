"""Offline tests for the NVD CVE API document loader."""

import json
from io import BytesIO
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from langchain_core.document_loaders import BaseLoader

from nvd_loader import NVDLoader


def _mock_api_response(records: list[dict]) -> BytesIO:
    """Build a byte stream matching the NVD API's top-level response shape."""
    return BytesIO(json.dumps({"vulnerabilities": records}).encode("utf-8"))


def test_load_parses_cve_with_product_and_cvss_metadata() -> None:
    """Map an enriched CVE response into one searchable LangChain document."""
    record = {
        "cve": {
            "id": "CVE-2024-12345",
            "published": "2024-01-02T00:00:00.000",
            "lastModified": "2024-01-03T00:00:00.000",
            "descriptions": [
                {"lang": "en", "value": "Example Server allows remote code execution."}
            ],
            "affected": [{"vendor": "example", "product": "server", "versions": []}],
            "metrics": {
                "cvssMetricV31": [
                    {
                        "type": "Primary",
                        "cvssData": {"baseScore": 9.8, "baseSeverity": "CRITICAL"},
                        "baseSeverity": "CRITICAL",
                    }
                ]
            },
        }
    }

    with patch("nvd_loader.urlopen", return_value=_mock_api_response([record])) as open_url:
        loader = NVDLoader(keyword="example server", max_cves=3)
        assert isinstance(loader, BaseLoader)
        documents = loader.load()

    assert len(documents) == 1
    document = documents[0]
    assert "CVE ID: CVE-2024-12345" in document.page_content
    assert "remote code execution" in document.page_content
    assert "example server" in document.page_content
    assert "CVSS 3.1: 9.8 (CRITICAL)" in document.page_content
    assert document.metadata == {
        "source": "nvd",
        "advisory_id": "CVE-2024-12345",
        "title": "CVE-2024-12345: Example Server allows remote code execution",
        "product": "example server",
        "severity": "CRITICAL",
        "cvss_score": 9.8,
        "cvss_version": "3.1",
        "published": "2024-01-02T00:00:00.000",
        "last_modified": "2024-01-03T00:00:00.000",
        "url": "https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
    }

    request = open_url.call_args.args[0]
    parsed = urlparse(request.full_url)
    assert parsed.path == "/rest/json/cves/2.0"
    assert parse_qs(parsed.query) == {"keywordSearch": ["example server"], "resultsPerPage": ["3"]}
    assert "apiKey" not in request.headers


def test_load_handles_cve_without_optional_cvss_or_product_fields() -> None:
    """Keep sparse CVE records loadable when enrichment fields are absent."""
    record = {
        "cve": {
            "id": "CVE-2025-54321",
            "published": "2025-02-01T00:00:00.000",
            "lastModified": "2025-02-02T00:00:00.000",
            "descriptions": [{"lang": "en", "value": "A vulnerability with limited details."}],
        }
    }

    with patch("nvd_loader.urlopen", return_value=_mock_api_response([record])):
        document = NVDLoader(keyword="limited details").load()[0]

    assert "CVE ID: CVE-2025-54321" in document.page_content
    assert "A vulnerability with limited details." in document.page_content
    assert document.metadata["source"] == "nvd"
    assert document.metadata["advisory_id"] == "CVE-2025-54321"
    assert document.metadata["title"].startswith("CVE-2025-54321:")
    assert document.metadata["published"] == "2025-02-01T00:00:00.000"
    assert document.metadata["last_modified"] == "2025-02-02T00:00:00.000"
    assert document.metadata["url"] == "https://nvd.nist.gov/vuln/detail/CVE-2025-54321"
    assert "product" not in document.metadata
    assert "severity" not in document.metadata
    assert "cvss_score" not in document.metadata
    assert "cvss_version" not in document.metadata
