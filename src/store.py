from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed, LocalEmbedder, OpenAIEmbedder
from .models import Document
import chromadb


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        client = chromadb.Client()

        try:
            client.delete_collection(name=self._collection_name)
        except Exception:
            pass
            
        self._collection = client.get_or_create_collection(name=self._collection_name)

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """
        Build a normalized stored record for one document.

        Expected normalized shape:
        {
            "id": str,
            "content": str,
            "embedding": list[float],
            "metadata": dict[str, Any],
        }
        """
        content = getattr(doc, "content", "") or ""
        metadata = dict(getattr(doc, "metadata", {}) or {})

        doc_id = getattr(doc, "id", None)
        if doc_id is None:
            doc_id = metadata.get("doc_id")

        if doc_id is not None:
            metadata.setdefault("doc_id", str(doc_id))

        record_id = f"{metadata.get('doc_id', 'doc')}-{self._next_index}"
        embedding = self._embedding_fn(content)

        return {
            "id": record_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
        }

    def _search_records(
        self, query: str, records: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """
        Run in-memory similarity search over provided records.
        """
        query_embedding = self._embedding_fn(query)

        scored: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            embedding = record.get("embedding", [])
            score = _dot(query_embedding, embedding)
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, record in scored[:top_k]:
            results.append(
                {
                    "id": record["id"],
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "score": score,
                }
            )
        return results

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        records = [self._make_record(doc) for doc in docs]

        if self._use_chroma and self._collection is not None:
            self._collection.add(
                ids=[record["id"] for record in records],
                documents=[record["content"] for record in records],
                embeddings=[record["embedding"] for record in records],
                metadatas=[record["metadata"] for record in records],
            )
        else:
            self._store.extend(records)

        self._next_index += len(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if top_k <= 0:
            return []

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            ids = (result.get("ids") or [[]])[0]
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]

            output: list[dict[str, Any]] = []
            for idx, doc_id in enumerate(ids):
                distance = distances[idx] if idx < len(distances) else None
                score = -distance if distance is not None else None
                output.append(
                    {
                        "id": doc_id,
                        "content": documents[idx] if idx < len(documents) else "",
                        "metadata": metadatas[idx] if idx < len(metadatas) else {},
                        "score": score,
                    }
                )
            return output

        return self._search_records(query=query, records=self._store, top_k=top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return int(self._collection.count())
        return len(self._store)

    def search_with_filter(
        self, query: str, top_k: int = 3, metadata_filter: dict = None
    ) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        metadata_filter = metadata_filter or {}

        if top_k <= 0:
            return []

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if metadata_filter:
                kwargs["where"] = metadata_filter

            result = self._collection.query(**kwargs)

            ids = (result.get("ids") or [[]])[0]
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]

            output: list[dict[str, Any]] = []
            for idx, doc_id in enumerate(ids):
                distance = distances[idx] if idx < len(distances) else None
                score = -distance if distance is not None else None
                output.append(
                    {
                        "id": doc_id,
                        "content": documents[idx] if idx < len(documents) else "",
                        "metadata": metadatas[idx] if idx < len(metadatas) else {},
                        "score": score,
                    }
                )
            return output

        filtered_records = self._store
        if metadata_filter:
            filtered_records = [
                record
                for record in self._store
                if all(record.get("metadata", {}).get(k) == v for k, v in metadata_filter.items())
            ]

        return self._search_records(query=query, records=filtered_records, top_k=top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            existing = self._collection.get(where={"doc_id": doc_id}, include=[])
            ids = existing.get("ids", []) if existing else []
            if not ids:
                return False

            self._collection.delete(where={"doc_id": doc_id})
            return True

        before = len(self._store)
        self._store = [
            record
            for record in self._store
            if record.get("metadata", {}).get("doc_id") != doc_id
        ]
        return len(self._store) < before