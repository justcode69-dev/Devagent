"""Language-aware chunking: splits on class/function boundaries first."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

_LANG = {".py": Language.PYTHON, ".js": Language.JS, ".jsx": Language.JS,
         ".ts": Language.TS, ".tsx": Language.TS, ".md": Language.MARKDOWN}


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str


def chunk_file(path: str, content: str, chunk_size: int = 1200, overlap: int = 150) -> list[Chunk]:
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    lang = _LANG.get(ext)
    if lang is not None:
        splitter = RecursiveCharacterTextSplitter.from_language(
            lang, chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True)
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True)
    chunks = []
    for doc in splitter.create_documents([content]):
        start = content.count("\n", 0, doc.metadata["start_index"]) + 1
        end = start + doc.page_content.count("\n")
        chunks.append(Chunk(path=path, start_line=start, end_line=end, text=doc.page_content))
    return chunks
