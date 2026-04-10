from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks



class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # TODO: split into sentences, group into chunks
        # raise NotImplementedError("Implement SentenceChunker.chunk")
        sentence_endings = re.compile(r'(?<=[.!?])\s+|\.\n')
        sentences = sentence_endings.split(text.strip())
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + self.max_sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # TODO: implement recursive splitting strategy
        # raise NotImplementedError("Implement RecursiveChunker.chunk")
        
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # TODO: recursive helper used by RecursiveChunker.chunk
        # raise NotImplementedError("Implement RecursiveChunker._split")
        if len(current_text) <= self.chunk_size or not remaining_separators:
            return [current_text.strip()]
        separator = remaining_separators[0]
        parts = current_text.split(separator)
        chunks = []
        for part in parts:
            chunks.extend(self._split(part, remaining_separators[1:]))
        return chunks
    
import re


class DocumentStructureChunker:
    """
    Split text by document structure first, then by size if needed.

    Rules:
        - Prefer splitting on markdown-style headings (#, ##, ###, ...)
        - Keep each heading with its section content
        - If a section is still too large, split it into smaller chunks
        - If no headings are found, fall back to fixed-size chunking
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_by_headings(self, text: str) -> list[str]:
        """
        Split markdown text into sections based on headings.
        Each section keeps its heading attached to its content.
        """
        lines = text.splitlines()
        sections: list[str] = []
        current_section: list[str] = []

        heading_pattern = re.compile(r"^\s{0,3}#{1,6}\s+")

        for line in lines:
            if heading_pattern.match(line):
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if section_text:
                        sections.append(section_text)
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text:
                sections.append(section_text)

        return sections

    def _split_large_section(self, section: str) -> list[str]:
        """
        Split one oversized section into smaller chunks using sliding-window logic.
        """
        if len(section) <= self.chunk_size:
            return [section]

        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        step = self.chunk_size - self.overlap
        chunks: list[str] = []

        for start in range(0, len(section), step):
            chunk = section[start : start + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
            if start + self.chunk_size >= len(section):
                break

        return chunks

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text]

        sections = self._split_by_headings(text)

        # Fallback nếu tài liệu không có heading
        if len(sections) == 1 and sections[0] == text.strip():
            return self._split_large_section(text)

        chunks: list[str] = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self._split_large_section(section))

        return chunks
    


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    cosine_similarity = dot(a, b) / (||a|| * ||b||)
    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    # raise NotImplementedError("Implement compute_similarity")
    dot_product = _dot(vec_a, vec_b)
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        # raise NotImplementedError("Implement ChunkingStrategyComparator.compare")
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)
        document_structure_chunker = DocumentStructureChunker(chunk_size=chunk_size, overlap=50)
        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)
        document_structure_chunks = document_structure_chunker.chunk(text)
        return {
            "fixed_size": {
                "chunks": fixed_chunks,
                "count": len(fixed_chunks),
                "avg_length": sum(len(chunk) for chunk in fixed_chunks) / len(fixed_chunks) if fixed_chunks else 0,
            },
            "by_sentences": {
                "chunks": sentence_chunks,
                "count": len(sentence_chunks),
                "avg_length": sum(len(chunk) for chunk in sentence_chunks) / len(sentence_chunks) if sentence_chunks else 0,
            },
            "recursive": {
                "chunks": recursive_chunks,
                "count": len(recursive_chunks),
                "avg_length": sum(len(chunk) for chunk in recursive_chunks) / len(recursive_chunks) if recursive_chunks else 0,
            },
            "document_structure": {
                "chunks": document_structure_chunks,
                "count": len(document_structure_chunks),
                "avg_length": sum(len(chunk) for chunk in document_structure_chunks) / len(document_structure_chunks) if document_structure_chunks else 0,
            }
        }
