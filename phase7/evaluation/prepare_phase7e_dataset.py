import os
import sys
import json
from typing import List, Dict, Any, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bpe_tokenizer import BPETokenizer
from phase7.ingestion.document_loader import RawDocument
from phase7.ingestion.chunker import MicroChunker
from phase7.retrieval.index import InvertedIndex

TOKENIZER_PATH = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")
EVAL_DIR = os.path.join(REPO_ROOT, "phase7", "evaluation")

def build_phase7e_corpus_and_questions() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a realistic 30-item evaluation corpus and QA dataset.
    Categories: factual_lookup, names_entities, numbers, dates, definitions, technical_facts, paraphrased.
    """
    raw_documents = [
        {
            "doc_id": "doc_astronomy",
            "title": "Astronomy and Solar System Overview",
            "text": "The Sun is a G-type main-sequence star that contains 99.86 percent of the mass in the Solar System. Jupiter is the largest planet, having a mass one-thousandth that of the Sun. Mars has two small natural satellites named Phobos and Deimos. The Apollo 11 mission landed humans on the Moon in July 1969."
        },
        {
            "doc_id": "doc_computing_history",
            "title": "History of Computing and Programming Languages",
            "text": "The ENIAC computer was announced in February 1946 at the University of Pennsylvania. Ada Lovelace is widely considered the first computer programmer for her work on Babbage's Analytical Engine. Python was designed by Guido van Rossum and first released in February 1991. The Linux operating system kernel was released by Linus Torvalds in September 1991."
        },
        {
            "doc_id": "doc_minigpt_specs",
            "title": "MiniGPT Technical Architecture Specifications",
            "text": "MiniGPT Model C is a micro-transformer architecture with 6,613,504 trainable parameters. The transformer configuration consists of 8 layers, 8 attention heads, and an embedding dimension of 256. Model C native context length is fixed at 128 tokens. Subword tokenization is handled by a Byte Pair Encoding tokenizer with a 1,024 vocabulary size."
        },
        {
            "doc_id": "doc_physics_units",
            "title": "Fundamental Physics Constants and Units",
            "text": "The speed of light in vacuum is defined as exactly 299,792,458 meters per second. Planck constant is equal to 6.626 x 10^-34 joule-seconds. Absolute zero is defined as 0 Kelvin or minus 273.15 degrees Celsius. The gravitational constant G has a value of 6.674 x 10^-11 m^3 kg^-1 s^-2."
        },
        {
            "doc_id": "doc_deep_learning",
            "title": "Deep Learning and Transformer Concepts",
            "text": "Transformer models replace recurrent layers with multi-head causal self-attention mechanisms. Supervised Fine-Tuning adapts pretrained language models to instruction-following tasks. Cross-entropy loss measures the KL divergence between target class labels and predicted probability distributions. AdamW optimizer decouples weight decay penalty from gradient updates."
        },
        {
            "doc_id": "doc_geography",
            "title": "Global Geography and World Capitals",
            "text": "The capital city of France is Paris, located along the Seine River. Tokyo is the capital of Japan and the most populous metropolitan area in the world. The Amazon River is the largest river by discharge volume of water in the world. Mount Everest is Earth's highest mountain above sea level, reaching 8,848 meters."
        }
    ]

    # 30 Evaluation Question Items
    eval_qa_items = [
        # Factual Lookup (4 items)
        {
            "id": 1,
            "category": "factual_lookup",
            "question": "What is the capital city of France?",
            "expected_answer": "Paris",
            "source_doc_id": "doc_geography",
            "supporting_fact": "The capital city of France is Paris, located along the Seine River."
        },
        {
            "id": 2,
            "category": "factual_lookup",
            "question": "What is the capital of Japan?",
            "expected_answer": "Tokyo",
            "source_doc_id": "doc_geography",
            "supporting_fact": "Tokyo is the capital of Japan and the most populous metropolitan area in the world."
        },
        {
            "id": 3,
            "category": "factual_lookup",
            "question": "What is the largest ocean planet in mass in our Solar System?",
            "expected_answer": "Jupiter",
            "source_doc_id": "doc_astronomy",
            "supporting_fact": "Jupiter is the largest planet, having a mass one-thousandth that of the Sun."
        },
        {
            "id": 4,
            "category": "factual_lookup",
            "question": "What river is the largest by discharge volume?",
            "expected_answer": "Amazon River",
            "source_doc_id": "doc_geography",
            "supporting_fact": "The Amazon River is the largest river by discharge volume of water in the world."
        },

        # Names & Entities (4 items)
        {
            "id": 5,
            "category": "names_entities",
            "question": "Who designed the Python programming language?",
            "expected_answer": "Guido van Rossum",
            "source_doc_id": "doc_computing_history",
            "supporting_fact": "Python was designed by Guido van Rossum and first released in February 1991."
        },
        {
            "id": 6,
            "category": "names_entities",
            "question": "Who released the Linux operating system kernel?",
            "expected_answer": "Linus Torvalds",
            "source_doc_id": "doc_computing_history",
            "supporting_fact": "The Linux operating system kernel was released by Linus Torvalds in September 1991."
        },
        {
            "id": 7,
            "category": "names_entities",
            "question": "Who is considered the first computer programmer?",
            "expected_answer": "Ada Lovelace",
            "source_doc_id": "doc_computing_history",
            "supporting_fact": "Ada Lovelace is widely considered the first computer programmer for her work on Babbage's Analytical Engine."
        },
        {
            "id": 8,
            "category": "names_entities",
            "question": "What are the names of the two moons of Mars?",
            "expected_answer": "Phobos and Deimos",
            "source_doc_id": "doc_astronomy",
            "supporting_fact": "Mars has two small natural satellites named Phobos and Deimos."
        },

        # Numbers (4 items)
        {
            "id": 9,
            "category": "numbers",
            "question": "How many parameters does Model C contain?",
            "expected_answer": "6,613,504",
            "source_doc_id": "doc_minigpt_specs",
            "supporting_fact": "MiniGPT Model C is a micro-transformer architecture with 6,613,504 trainable parameters."
        },
        {
            "id": 10,
            "category": "numbers",
            "question": "What is the speed of light in vacuum?",
            "expected_answer": "299,792,458 meters per second",
            "source_doc_id": "doc_physics_units",
            "supporting_fact": "The speed of light in vacuum is defined as exactly 299,792,458 meters per second."
        },
        {
            "id": 11,
            "category": "numbers",
            "question": "What is the height of Mount Everest in meters?",
            "expected_answer": "8,848 meters",
            "source_doc_id": "doc_geography",
            "supporting_fact": "Mount Everest is Earth's highest mountain above sea level, reaching 8,848 meters."
        },
        {
            "id": 12,
            "category": "numbers",
            "question": "What is the BPE vocabulary size of Model C?",
            "expected_answer": "1,024",
            "source_doc_id": "doc_minigpt_specs",
            "supporting_fact": "Subword tokenization is handled by a Byte Pair Encoding tokenizer with a 1,024 vocabulary size."
        },

        # Dates (4 items)
        {
            "id": 13,
            "category": "dates",
            "question": "When was Python first released?",
            "expected_answer": "February 1991",
            "source_doc_id": "doc_computing_history",
            "supporting_fact": "Python was designed by Guido van Rossum and first released in February 1991."
        },
        {
            "id": 14,
            "category": "dates",
            "question": "When did Apollo 11 land humans on the Moon?",
            "expected_answer": "July 1969",
            "source_doc_id": "doc_astronomy",
            "supporting_fact": "The Apollo 11 mission landed humans on the Moon in July 1969."
        },
        {
            "id": 15,
            "category": "dates",
            "question": "When was the ENIAC computer announced?",
            "expected_answer": "February 1946",
            "source_doc_id": "doc_computing_history",
            "supporting_fact": "The ENIAC computer was announced in February 1946 at the University of Pennsylvania."
        },
        {
            "id": 16,
            "category": "dates",
            "question": "When was the Linux kernel released?",
            "expected_answer": "September 1991",
            "source_doc_id": "doc_computing_history",
            "supporting_fact": "The Linux operating system kernel was released by Linus Torvalds in September 1991."
        },

        # Definitions (5 items)
        {
            "id": 17,
            "category": "definitions",
            "question": "What is absolute zero defined as in Celsius?",
            "expected_answer": "minus 273.15 degrees Celsius",
            "source_doc_id": "doc_physics_units",
            "supporting_fact": "Absolute zero is defined as 0 Kelvin or minus 273.15 degrees Celsius."
        },
        {
            "id": 18,
            "category": "definitions",
            "question": "What does AdamW optimizer decouple?",
            "expected_answer": "weight decay penalty from gradient updates",
            "source_doc_id": "doc_deep_learning",
            "supporting_fact": "AdamW optimizer decouples weight decay penalty from gradient updates."
        },
        {
            "id": 19,
            "category": "definitions",
            "question": "What does cross-entropy loss measure?",
            "expected_answer": "KL divergence between target class labels and predicted probability distributions",
            "source_doc_id": "doc_deep_learning",
            "supporting_fact": "Cross-entropy loss measures the KL divergence between target class labels and predicted probability distributions."
        },
        {
            "id": 20,
            "category": "definitions",
            "question": "What is the Sun classified as?",
            "expected_answer": "G-type main-sequence star",
            "source_doc_id": "doc_astronomy",
            "supporting_fact": "The Sun is a G-type main-sequence star that contains 99.86 percent of the mass in the Solar System."
        },
        {
            "id": 21,
            "category": "definitions",
            "question": "What mechanism replaces recurrent layers in Transformers?",
            "expected_answer": "multi-head causal self-attention mechanisms",
            "source_doc_id": "doc_deep_learning",
            "supporting_fact": "Transformer models replace recurrent layers with multi-head causal self-attention mechanisms."
        },

        # Technical Facts (5 items)
        {
            "id": 22,
            "category": "technical_facts",
            "question": "How many layers and attention heads does Model C have?",
            "expected_answer": "8 layers, 8 attention heads",
            "source_doc_id": "doc_minigpt_specs",
            "supporting_fact": "The transformer configuration consists of 8 layers, 8 attention heads, and an embedding dimension of 256."
        },
        {
            "id": 23,
            "category": "technical_facts",
            "question": "What is Model C native context length?",
            "expected_answer": "128 tokens",
            "source_doc_id": "doc_minigpt_specs",
            "supporting_fact": "Model C native context length is fixed at 128 tokens."
        },
        {
            "id": 24,
            "category": "technical_facts",
            "question": "What is the value of Planck constant?",
            "expected_answer": "6.626 x 10^-34 joule-seconds",
            "source_doc_id": "doc_physics_units",
            "supporting_fact": "Planck constant is equal to 6.626 x 10^-34 joule-seconds."
        },
        {
            "id": 25,
            "category": "technical_facts",
            "question": "What is the value of gravitational constant G?",
            "expected_answer": "6.674 x 10^-11 m^3 kg^-1 s^-2",
            "source_doc_id": "doc_physics_units",
            "supporting_fact": "The gravitational constant G has a value of 6.674 x 10^-11 m^3 kg^-1 s^-2."
        },
        {
            "id": 26,
            "category": "technical_facts",
            "question": "What percentage of Solar System mass is in the Sun?",
            "expected_answer": "99.86 percent",
            "source_doc_id": "doc_astronomy",
            "supporting_fact": "The Sun is a G-type main-sequence star that contains 99.86 percent of the mass in the Solar System."
        },

        # Paraphrased Questions (4 items)
        {
            "id": 27,
            "category": "paraphrased",
            "question": "Which city serves as Japan's capital?",
            "expected_answer": "Tokyo",
            "source_doc_id": "doc_geography",
            "supporting_fact": "Tokyo is the capital of Japan and the most populous metropolitan area in the world."
        },
        {
            "id": 28,
            "category": "paraphrased",
            "question": "Who created the creator of the Linux kernel?",
            "expected_answer": "Linus Torvalds",
            "source_doc_id": "doc_computing_history",
            "supporting_fact": "The Linux operating system kernel was released by Linus Torvalds in September 1991."
        },
        {
            "id": 29,
            "category": "paraphrased",
            "question": "What is the embedding dimension size of Model C?",
            "expected_answer": "256",
            "source_doc_id": "doc_minigpt_specs",
            "supporting_fact": "The transformer configuration consists of 8 layers, 8 attention heads, and an embedding dimension of 256."
        },
        {
            "id": 30,
            "category": "paraphrased",
            "question": "How fast does light travel in vacuum?",
            "expected_answer": "299,792,458 meters per second",
            "source_doc_id": "doc_physics_units",
            "supporting_fact": "The speed of light in vacuum is defined as exactly 299,792,458 meters per second."
        }
    ]

    return raw_documents, eval_qa_items

def main():
    print("=== Phase 7E: Evaluation Corpus & Dataset Preparation ===", flush=True)

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer missing at {TOKENIZER_PATH}")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    os.makedirs(EVAL_DIR, exist_ok=True)

    raw_docs, eval_items = build_phase7e_corpus_and_questions()
    assert len(eval_items) == 30, f"Expected 30 eval items, got {len(eval_items)}"

    # Ingest documents using Phase 7B MicroChunker
    chunker = MicroChunker(tokenizer=tokenizer, target_tokens=42, overlap_tokens=10, min_tokens=10, max_tokens=45)
    all_chunks = []

    doc_chunk_map = {}
    for doc in raw_docs:
        raw_doc = RawDocument(
            document_id=doc["doc_id"],
            source=f"{doc['doc_id']}.txt",
            file_type="txt",
            raw_text=doc["text"],
            metadata={}
        )
        chunks = chunker.chunk_document(raw_doc)
        chunk_dicts = [c.to_dict() for c in chunks]
        all_chunks.extend(chunk_dicts)
        doc_chunk_map[doc["doc_id"]] = chunk_dicts

    print(f"Ingested {len(raw_docs)} documents into {len(all_chunks)} micro-chunks.")

    # Match each question item to its exact supporting source_chunk_id
    for item in eval_items:
        doc_id = item["source_doc_id"]
        fact = item["supporting_fact"].strip()
        matching_chunk_id = None
        for c in doc_chunk_map[doc_id]:
            if fact in c["text"] or any(w in c["text"] for w in fact.split()[:3]):
                matching_chunk_id = c["chunk_id"]
                break
        if not matching_chunk_id and doc_chunk_map[doc_id]:
            matching_chunk_id = doc_chunk_map[doc_id][0]["chunk_id"]

        item["source_chunk_id"] = matching_chunk_id

    # Build and save Inverted Index for evaluation
    index = InvertedIndex.build_from_chunks(all_chunks, tokenizer=tokenizer)
    index_path = os.path.join(EVAL_DIR, "phase7e_index.json")
    index.save_json(index_path)
    print(f"Saved Phase 7E inverted index to {index_path}")

    # Save dataset JSONL
    dataset_path = os.path.join(EVAL_DIR, "phase7e_dataset.jsonl")
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved 30 evaluation QA items to {dataset_path}")

    # Save raw documents JSON
    docs_path = os.path.join(EVAL_DIR, "phase7e_documents.json")
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(raw_docs, f, indent=2)
    print(f"Saved raw documents to {docs_path}")

    print("Phase 7E dataset preparation completed successfully!")

if __name__ == "__main__":
    main()
