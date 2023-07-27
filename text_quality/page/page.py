from pathlib import Path
from typing import List
from pagexml.model.physical_document_model import PageXMLScan
from pagexml.parser import parse_pagexml_file
from ..settings import LINE_SEPARATOR


class Page:
    """A wrapper around a PageXML file."""

    def __init__(self, page_doc: PageXMLScan) -> None:
        self._page_doc = page_doc

    @property
    def id(self) -> str:
        """The page id."""
        return self._page_doc.id

    def lines(self) -> List[str]:
        """Return lines from page."""
        return [
            line.text for line in self._page_doc.get_lines() if line.text is not None
        ]

    def get_text(self):
        """Get the entire text of the page."""
        return LINE_SEPARATOR.join(self.lines())

    @classmethod
    def from_file(cls, file: Path):
        return cls(parse_pagexml_file(file))
