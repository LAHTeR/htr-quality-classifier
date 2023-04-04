import os
from pathlib import Path
from typing import List
from pagexml.model.physical_document_model import PageXMLScan
from pagexml.parser import parse_pagexml_file


class Page:
    def __init__(self, page_doc: PageXMLScan) -> None:
        self._page_doc = page_doc

    def _lines(self) -> List[str]:
        return [
            line.text for line in self._page_doc.get_lines() if line.text is not None
        ]

    def get_text(self):
        return os.linesep.join(self._lines())

    @classmethod
    def from_file(cls, file: Path):
        return cls(parse_pagexml_file(file))
