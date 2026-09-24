"""Unit tests for the local Wikipedia loader."""

from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.document_loaders import BaseLoader
from wikipedia_loader import WikipediaLoader


def test_load_returns_langchain_documents_from_wikipedia_pages() -> None:
    """Verify BaseLoader.load returns mocked Wikipedia pages as documents."""
    # Prepare fake Wikipedia pages so the test does not need network access.
    pages = {
        "LangChain": SimpleNamespace(
            title="LangChain",
            content="LangChain is a framework.",
            summary="A framework.",
            url="https://en.wikipedia.org/wiki/LangChain",
        ),
        "Python": SimpleNamespace(
            title="Python (programming language)",
            content="Python is a programming language.",
            summary="A programming language.",
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        ),
    }

    with (
        patch("wikipedia_loader.wikipedia.search", return_value=["LangChain", "Python"]) as search,
        patch("wikipedia_loader.wikipedia.page", side_effect=lambda title, **_: pages[title]) as page,
    ):
        # Load documents through the real loader while Wikipedia calls are mocked.
        documents = WikipediaLoader(query="LangChain", load_max_docs=2).load()

    # Verify the search inputs, loaded content, and document metadata.
    search.assert_called_once_with("LangChain", results=2)
    assert page.call_count == 2
    assert [document.page_content for document in documents] == [
        "LangChain is a framework.",
        "Python is a programming language.",
    ]
    assert documents[0].metadata == {
        "source": "https://en.wikipedia.org/wiki/LangChain",
        "title": "LangChain",
        "summary": "A framework.",
    }


def test_lazy_load_yields_documents_lazily() -> None:
    """Verify WikipediaLoader follows BaseLoader and defers requests until iterated."""
    loader = WikipediaLoader(query="LangChain", load_max_docs=1)
    fake_page = SimpleNamespace(
        title="LangChain",
        content="LangChain is a framework.",
        summary="A framework.",
        url="https://en.wikipedia.org/wiki/LangChain",
    )

    assert isinstance(loader, BaseLoader)
    with (
        patch("wikipedia_loader.wikipedia.search", return_value=["LangChain"]) as search,
        patch("wikipedia_loader.wikipedia.page", return_value=fake_page) as page,
    ):
        documents = loader.lazy_load()
        search.assert_not_called()
        page.assert_not_called()

        document = next(documents)

    search.assert_called_once_with("LangChain", results=1)
    page.assert_called_once_with("LangChain", auto_suggest=False)
    assert document.page_content == "LangChain is a framework."
    assert document.metadata["source"] == "https://en.wikipedia.org/wiki/LangChain"
