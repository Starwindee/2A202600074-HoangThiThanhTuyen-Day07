from pathlib import Path
from src.models import Document
from src.store import EmbeddingStore
from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker, DocumentStructureChunker, ChunkingStrategyComparator
from src.embeddings import _mock_embed, LocalEmbedder, OpenAIEmbedder
import uuid

# ============= BƯỚC 1: Load Docs =============
def load_sample_docs() -> list[Document]:
    """Load sample documents from the 'sample_docs' directory."""
    sample_dir = Path("./data")
    files = list(sample_dir.glob("*.md"))
    
    docs = []
    for file_path in files:
        p = Path(file_path)
        if p.exists():
            content = p.read_text(encoding="utf-8")
            docs.append(Document(
                id=p.stem,  # filename without extension
                content=content,
                metadata={"source": file_path}
            ))
    return docs

# ============= BƯỚC 2: Define Benchmark Queries =============
BENCHMARK_QUERIES = [
    {
        "query": "Hướng dẫn yêu cầu xuất hóa đơn VAT và cách kiểm tra hóa đơn với các chuyến xe Xanh SM",
        "gold_answer": "NaN"
    },
    {
        "query": "Làm sao khi hành khách để quên đồ trên xe?",
        "gold_answer": "NaN"
    },
    {
        "query": "Ngoài lương thưởng, tôi còn được hưởng chính sách gì nữa?",
        "gold_answer": "NaN"
    },
    {
        "query": "Tôi muốn đặt chuyến giao đồ ăn trên ứng dụng",
        "gold_answer": "NaN"
    },
    {
        "query": "Quán có rating trên Google và muốn đồng bộ về Ứng dụng Xanh SM",
        "gold_answer": "NaN"
    },
]

# ============= BƯỚC 3: Define Strategies =============
STRATEGIES = {
    "fixed_size_300": {
        "chunker": FixedSizeChunker(chunk_size=300, overlap=60),
        "name": "Fixed-Size (300 chars, 60 overlap)"
    },
    "sentence_3": {
        "chunker": SentenceChunker(max_sentences_per_chunk=3),
        "name": "Sentence-Based (3 sentences)"
    },
    "recursive_350": {
        "chunker": RecursiveChunker(chunk_size=350),
        "name": "Recursive (350 chars)"
    },
    "document_structure_200": {
    "chunker": DocumentStructureChunker(chunk_size=200, overlap=50),
        "name": "Document Structure (headings + size fallback)"
    }
}

# ============= BƯỚC 4: Run Benchmark =============
def run_benchmark():
    """Chạy benchmark cho tất cả strategy và query"""
    docs = load_sample_docs()
    if not docs:
        print("No docs loaded!")
        return
    
    print(f"Loaded {len(docs)} documents\n")
    
    results = {}
    
    for strategy_key, strategy_info in STRATEGIES.items():
        print(f"\n{'='*80}")
        print(f"STRATEGY: {strategy_info['name']}")
        print(f"{'='*80}")
        
        chunker = strategy_info['chunker']
        
        # Tạo store và chunk tất cả docs
        store = EmbeddingStore(
            collection_name=f"test_{strategy_key}_{uuid.uuid4().hex}",
            embedding_fn=LocalEmbedder()
        )
# Chunk tất cả documents
        all_chunk_docs = []
        for doc in docs:
            chunks = chunker.chunk(doc.content)
            for i, chunk in enumerate(chunks):
                chunk_docs = Document(
                    id=f"{doc.id}_chunk{i}",
                    content=chunk,
                    metadata={"doc_id": doc.id, "source": str(doc.metadata["source"])}
                )
                all_chunk_docs.append(chunk_docs)
        
        store.add_documents(all_chunk_docs)
        
        print(f"\nStore size: {store.get_collection_size()} chunks\n")
        
        strategy_results = {}
        
        # Test từng query
        for query_idx, q in enumerate(BENCHMARK_QUERIES, 1):
            query = q["query"]
            gold = q["gold_answer"]
            
            print(f"\n  Query {query_idx}: {query}")
            print(f"  Gold: {gold}\n")
            
            # Search top-3
            results_list = store.search(query, top_k=3)
            
            print(f"  Top-3 Results:")
            for rank, res in enumerate(results_list, 1):
                score = res.get("score", 0)
                content_preview = res["content"][:100].replace("\n", " ")
                print(f"    [{rank}] score={score:.3f}: {content_preview}...")
            
            strategy_results[query] = results_list
        
        results[strategy_key] = strategy_results
    
    # ============= BƯỚC 5: So Sánh Kết Quả =============
    print(f"\n\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    # ====== LƯU SCORE ======
    strategy_avg_scores = {key: [] for key in STRATEGIES.keys()}

    print("Top Score for Each Query by Strategy:")
    print(f"{'Query':<40} | {'Fixed-Size':<15} | {'Sentence':<15} | {'Recursive':<15} | {'Document Structure':<20}")
    print("-" * 90)

    for q in BENCHMARK_QUERIES:
        query = q["query"]
        query_short = query[:35] + "..."

        scores = {}

        for strategy_key in STRATEGIES.keys():
            if query in results[strategy_key]:
                top_result = results[strategy_key][query]
                top_score = top_result[0].get("score", 0) if top_result else 0

                scores[strategy_key] = top_score
                strategy_avg_scores[strategy_key].append(top_score)

        print(
            f"{query_short:<40} | "
            f"{scores.get('fixed_size_300', 0):<15.3f} | "
            f"{scores.get('sentence_3', 0):<15.3f} | "
            f"{scores.get('recursive_350', 0):<15.3f}"
            f" | {scores.get('document_structure_200', 0):<20.3f}"
        )

    # ====== TÍNH AVERAGE ======
    print("\n" + "=" * 80)
    print("AVERAGE SCORE PER STRATEGY")
    print("=" * 80)

    avg_results = {}

    for strategy, score_list in strategy_avg_scores.items():
        if score_list:
            avg_score = sum(score_list) / len(score_list)
        else:
            avg_score = 0

        avg_results[strategy] = avg_score

        print(f"{strategy:<20}: {avg_score:.4f}")

    # ====== WIN STRATEGY ======
    best_strategy = max(avg_results.items(), key=lambda x: x[1])[0]

    print("\n" + "=" * 80)
    print("WINNER STRATEGY")
    print("=" * 80)

    print(f"🏆 Best Strategy: {best_strategy}")
    print(f"Score: {avg_results[best_strategy]:.4f}")

if __name__ == "__main__":
    run_benchmark()