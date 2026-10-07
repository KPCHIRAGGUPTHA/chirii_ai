import os
import sys
import json
import random
from typing import List, Dict, Any, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bpe_tokenizer import BPETokenizer

TOKENIZER_PATH = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")
OUTPUT_DIR = os.path.join(REPO_ROOT, "phase7", "data", "sft")

def generate_80_rag_examples() -> List[Dict[str, Any]]:
    """
    Generate exactly 80 synthetic, factual, context-grounded RAG QA pairs.
    5 categories x 16 examples each.
    """
    raw_data = []

    # Category 1: direct_lookup (16 examples)
    direct_lookup = [
        ("The capital city of France is Paris.", "What is the capital of France?", "Paris"),
        ("The capital of Japan is Tokyo.", "What is the capital of Japan?", "Tokyo"),
        ("The currency of the United Kingdom is the Pound Sterling.", "What is the currency of the UK?", "Pound Sterling"),
        ("The largest ocean on Earth is the Pacific Ocean.", "What is the largest ocean on Earth?", "Pacific Ocean"),
        ("The atomic symbol for gold is Au.", "What atomic symbol for gold?", "Au"),
        ("The chemical symbol for water is H2O.", "What chemical symbol for water?", "H2O"),
        ("The primary language spoken in Brazil is Portuguese.", "What language is spoken in Brazil?", "Portuguese"),
        ("The highest mountain above sea level is Mount Everest.", "What is the highest mountain?", "Mount Everest"),
        ("The capital city of Italy is Rome.", "What is the capital of Italy?", "Rome"),
        ("The symbol for sodium on the periodic table is Na.", "What symbol for sodium?", "Na"),
        ("The official currency of Japan is the Yen.", "What official currency of Japan?", "Yen"),
        ("The capital city of Spain is Madrid.", "What is the capital of Spain?", "Madrid"),
        ("The largest planet in our solar system is Jupiter.", "What is the largest planet?", "Jupiter"),
        ("The speed of sound in air at room temp is 343 meters per second.", "What is the speed of sound?", "343 meters per second"),
        ("The main component of Earth's atmosphere is Nitrogen.", "What main component of air?", "Nitrogen"),
        ("The capital of Canada is Ottawa.", "What is the capital of Canada?", "Ottawa")
    ]

    # Category 2: numbers (16 examples)
    numbers = [
        ("Model C contains 6,613,504 parameters.", "How many parameters in Model C?", "6,613,504"),
        ("The speed of light is 299,792,458 meters per second.", "What is the speed of light?", "299,792,458 meters per second"),
        ("The baseline context window size is 128 tokens.", "What context window size?", "128 tokens"),
        ("There are 8 transformer layers in Model C.", "How many transformer layers in Model C?", "8"),
        ("The vocabulary size of the BPE tokenizer is 1,024.", "What BPE vocabulary size?", "1,024"),
        ("The embedding dimension of Model C is 256.", "What embedding dimension in Model C?", "256"),
        ("There are 8 attention heads per block in Model C.", "How many attention heads per block?", "8"),
        ("The freezing baseline git commit SHA is 7bbbd6b.", "What baseline commit SHA?", "7bbbd6b"),
        ("The total number of full test suite tests is 151.", "How many tests in full suite?", "151"),
        ("The freezing baseline test perplexity was 19.9101.", "What baseline test perplexity?", "19.9101"),
        ("The fine-tuned SFT test perplexity was 13.5777.", "What SFT test perplexity?", "13.5777"),
        ("The earth orbits the sun in 365 days.", "How many days for Earth orbit?", "365 days"),
        ("Water boils at 100 degrees Celsius.", "At what temperature water boils?", "100 degrees Celsius"),
        ("A standard chessboard has 64 squares.", "How many squares on chessboard?", "64"),
        ("An hour consists of 3,600 seconds.", "How many seconds in an hour?", "3,600"),
        ("Model C has 8 heads and 8 layers.", "How many layers in Model C?", "8")
    ]

    # Category 3: names (16 examples)
    names = [
        ("Python was created by Guido van Rossum.", "Who created Python?", "Guido van Rossum"),
        ("The C programming language was created by Dennis Ritchie.", "Who created C language?", "Dennis Ritchie"),
        ("Linux kernel was initially developed by Linus Torvalds.", "Who developed Linux kernel?", "Linus Torvalds"),
        ("The Java programming language was designed by James Gosling.", "Who designed Java?", "James Gosling"),
        ("The theory of general relativity was published by Albert Einstein.", "Who published general relativity?", "Albert Einstein"),
        ("The telephone was patented by Alexander Graham Bell.", "Who patented the telephone?", "Alexander Graham Bell"),
        ("World Wide Web was invented by Tim Berners-Lee.", "Who invented World Wide Web?", "Tim Berners-Lee"),
        ("The C++ language was developed by Bjarne Stroustrup.", "Who developed C++?", "Bjarne Stroustrup"),
        ("Git version control system was created by Linus Torvalds.", "Who created Git?", "Linus Torvalds"),
        ("The Mona Lisa painting was created by Leonardo da Vinci.", "Who painted Mona Lisa?", "Leonardo da Vinci"),
        ("Apple Inc was co-founded by Steve Jobs and Steve Wozniak.", "Who co-founded Apple Inc?", "Steve Jobs and Steve Wozniak"),
        ("The play Hamlet was written by William Shakespeare.", "Who wrote Hamlet?", "William Shakespeare"),
        ("The principia mathematica was written by Isaac Newton.", "Who wrote Principia Mathematica?", "Isaac Newton"),
        ("PyTorch was initiated by Adam Lerer and Soumith Chintala.", "Who initiated PyTorch?", "Adam Lerer and Soumith Chintala"),
        ("The MiniGPT project AI agent is named Antigravity.", "What name of AI agent?", "Antigravity"),
        ("The creator of Rust is Graydon Hoare.", "Who created Rust?", "Graydon Hoare")
    ]

    # Category 4: multi_word (16 examples)
    multi_word = [
        ("Transformer models rely heavily on causal self-attention.", "What Transformer models rely on?", "causal self-attention"),
        ("Supervised Fine-Tuning optimizes instruction-following performance.", "What SFT optimizes?", "instruction-following performance"),
        ("RAG bridges internal parametric memory with external data.", "What RAG bridges?", "internal parametric memory with external data"),
        ("Byte Pair Encoding learns frequent subword token merges.", "What BPE learns?", "frequent subword token merges"),
        ("BM25 computes term frequency and inverse document frequency.", "What BM25 computes?", "term frequency and inverse document frequency"),
        ("Cross entropy loss measures discrepancy between distributions.", "What cross entropy loss measures?", "discrepancy between distributions"),
        ("Layer normalization stabilizes training of neural networks.", "What layer normalization stabilizes?", "training of neural networks"),
        ("Gradient clipping prevents exploding gradient issues.", "What gradient clipping prevents?", "exploding gradient issues"),
        ("Learning rate warmup increases learning rate during initial steps.", "What learning rate warmup does?", "increases learning rate during initial steps"),
        ("Weight decay applies L2 regularization to parameters.", "What weight decay applies?", "L2 regularization"),
        ("Softmax function converts unnormalized logits into probabilities.", "What softmax converts logits into?", "probabilities"),
        ("AdamW optimizer decouples weight decay from updates.", "What AdamW optimizer decouples?", "weight decay from updates"),
        ("Inverted index speeds up lexical keyword search.", "What inverted index speeds up?", "lexical keyword search"),
        ("Cosine similarity measures the angle between vector representations.", "What cosine similarity measures?", "the angle between vector representations"),
        ("GELU activation provides smooth non-linear transformations.", "What GELU activation provides?", "smooth non-linear transformations"),
        ("Precision and recall evaluate retrieval performance.", "What precision and recall evaluate?", "retrieval performance")
    ]

    # Category 5: paraphrased (16 examples)
    paraphrased = [
        ("The capital of Germany is Berlin.", "Which city is Germany's capital?", "Berlin"),
        ("Photosynthesis allows plants to convert sunlight into energy.", "How plants make energy from sunlight?", "Photosynthesis"),
        ("Oxygen is produced by plants during photosynthesis.", "What gas do plants produce?", "Oxygen"),
        ("The moon is the natural satellite of Earth.", "What body orbits Earth naturally?", "The moon"),
        ("Penicillin was discovered accidentally by Alexander Fleming.", "Who discovered Penicillin?", "Alexander Fleming"),
        ("Gravity draws objects toward the center of mass.", "What force pulls objects to mass?", "Gravity"),
        ("Electrons carry a negative electrical charge.", "What has negative charge?", "Electrons"),
        ("Protons carry a positive electrical charge.", "What has positive charge?", "Protons"),
        ("Neutrons carry zero net electrical charge.", "What has neutral charge?", "Neutrons"),
        ("Dna stores hereditary genetic material in living organisms.", "Where genetic material is stored?", "Dna"),
        ("Chlorophyll gives leaves their green color.", "What gives green color to leaves?", "Chlorophyll"),
        ("Diamond is composed entirely of carbon atoms.", "What element constitutes diamond?", "carbon"),
        ("Mariana Trench is located in the Pacific Ocean.", "Where is Mariana Trench located?", "Pacific Ocean"),
        ("Compilers translate source code into machine instructions.", "What turns code to machine instructions?", "Compilers"),
        ("An octopus has three hearts.", "How many hearts an octopus has?", "three"),
        ("Mount Fuji is the tallest mountain in Japan.", "Which mountain is Japan's peak?", "Mount Fuji")
    ]

    item_id = 1

    for cat_name, item_list in [
        ("direct_lookup", direct_lookup),
        ("numbers", numbers),
        ("names", names),
        ("multi_word", multi_word),
        ("paraphrased", paraphrased),
    ]:
        for context, question, answer in item_list:
            raw_data.append({
                "id": item_id,
                "category": cat_name,
                "context": context,
                "question": question,
                "answer": answer
            })
            item_id += 1

    return raw_data

def format_rag_sft_prompt(context: str, question: str, answer: str = "") -> Tuple[str, str]:
    """
    Format RAG-SFT prompt:
    Prompt Prefix: 'Context:\n{c}\n\nQuestion:\n{q}\n\nAnswer:\n'
    Full Text:     'Context:\n{c}\n\nQuestion:\n{q}\n\nAnswer:\n{a}'
    """
    c = context.strip()
    q = question.strip()
    a = answer.strip()
    prompt_prefix = f"Context:\n{c}\n\nQuestion:\n{q}\n\nAnswer:\n"
    full_text = f"{prompt_prefix}{a}" if a else prompt_prefix
    return prompt_prefix, full_text

def main():
    print("=== Phase 7D-C: RAG-SFT Dataset Preparation & Validation ===", flush=True)

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_items = generate_80_rag_examples()
    assert len(raw_items) == 80, f"Expected 80 records, got {len(raw_items)}"

    cat_counts: Dict[str, int] = {}
    for r in raw_items:
        c = r["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    print("Category Breakdown (80 total):", cat_counts)
    for c, count in cat_counts.items():
        assert count == 16, f"Category {c} count {count} != 16"

    # Token length verification for all 80 items
    tokenized_items = []
    for item in raw_items:
        prefix, full_text = format_rag_sft_prompt(item["context"], item["question"], item["answer"])
        
        ctx_tokens = tokenizer.encode(item["context"])
        q_tokens = tokenizer.encode(item["question"])
        ans_tokens = tokenizer.encode(item["answer"])
        
        prefix_tokens = tokenizer.encode(prefix)
        full_tokens = tokenizer.encode(full_text)
        
        assert len(ctx_tokens) <= 42, f"Item {item['id']} ({item['category']}) context tokens {len(ctx_tokens)} > 42"
        assert len(q_tokens) <= 22, f"Item {item['id']} ({item['category']}) question tokens {len(q_tokens)} > 22"
        assert len(ans_tokens) <= 30, f"Item {item['id']} ({item['category']}) answer tokens {len(ans_tokens)} > 30"
        assert len(full_tokens) <= 128, f"Item {item['id']} ({item['category']}) total tokens {len(full_tokens)} > 128"

        tokenized_items.append({
            "id": item["id"],
            "category": item["category"],
            "context": item["context"],
            "question": item["question"],
            "answer": item["answer"],
            "prompt_text": prefix,
            "full_text": full_text,
            "ctx_token_len": len(ctx_tokens),
            "q_token_len": len(q_tokens),
            "ans_token_len": len(ans_tokens),
            "prefix_token_len": len(prefix_tokens),
            "total_token_len": len(full_tokens),
            "tokens": full_tokens
        })

    # Train / Val / Test Partition Strategy (64 Train / 8 Val / 8 Test)
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for item in tokenized_items:
        by_category.setdefault(item["category"], []).append(item)

    # Seed 42 for deterministic partition
    random.seed(42)
    for c in by_category:
        random.shuffle(by_category[c])

    # Slice items per category:
    # direct_lookup (16): 12 train, 2 val, 2 test
    # numbers (16): 12 train, 2 val, 2 test
    # names (16): 12 train, 2 val, 2 test
    # multi_word (16): 14 train, 1 val, 1 test
    # paraphrased (16): 14 train, 1 val, 1 test
    slice_specs = {
        "direct_lookup": (12, 2, 2),
        "numbers": (12, 2, 2),
        "names": (12, 2, 2),
        "multi_word": (14, 1, 1),
        "paraphrased": (14, 1, 1),
    }

    train_set, val_set, test_set = [], [], []

    for cat, (n_tr, n_va, n_te) in slice_specs.items():
        cat_items = by_category[cat]
        train_set.extend(cat_items[:n_tr])
        val_set.extend(cat_items[n_tr:n_tr+n_va])
        test_set.extend(cat_items[n_tr+n_va:n_tr+n_va+n_te])

    assert len(train_set) == 64, f"Train set size {len(train_set)} != 64"
    assert len(val_set) == 8, f"Val set size {len(val_set)} != 8"
    assert len(test_set) == 8, f"Test set size {len(test_set)} != 8"

    # Verify ID uniqueness across splits & no duplicates
    train_ids = {x["id"] for x in train_set}
    val_ids = {x["id"] for x in val_set}
    test_ids = {x["id"] for x in test_set}

    assert len(train_ids.intersection(val_ids)) == 0, "Duplicate IDs in train/val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Duplicate IDs in train/test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Duplicate IDs in val/test!"

    # Save JSONL files
    files_to_save = [
        ("rag_sft_all.jsonl", tokenized_items),
        ("rag_sft_train.jsonl", train_set),
        ("rag_sft_val.jsonl", val_set),
        ("rag_sft_test.jsonl", test_set),
    ]

    for fname, dataset in files_to_save:
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            for rec in dataset:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(dataset)} items to {fpath}")

    # Calculate token length and target masking statistics
    all_totals = [x["total_token_len"] for x in tokenized_items]
    all_prefix_lens = [x["prefix_token_len"] for x in tokenized_items]
    all_ans_lens = [x["ans_token_len"] for x in tokenized_items]

    total_tokens_sum = sum(all_totals)
    prefix_tokens_sum = sum(all_prefix_lens) # Masked tokens (-100)
    ans_tokens_sum = sum(all_ans_lens)       # Active target tokens

    masked_pct = round((prefix_tokens_sum / total_tokens_sum) * 100, 2)
    active_pct = round((ans_tokens_sum / total_tokens_sum) * 100, 2)

    stats = {
        "total_records": 80,
        "splits": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set)
        },
        "category_counts": cat_counts,
        "token_stats": {
            "min_total": min(all_totals),
            "max_total": max(all_totals),
            "mean_total": round(sum(all_totals) / 80.0, 2),
            "mean_context_tokens": round(sum(x["ctx_token_len"] for x in tokenized_items) / 80.0, 2),
            "mean_question_tokens": round(sum(x["q_token_len"] for x in tokenized_items) / 80.0, 2),
            "mean_answer_tokens": round(sum(all_ans_lens) / 80.0, 2),
        },
        "masking_stats": {
            "total_tokens_sum": total_tokens_sum,
            "masked_prompt_tokens_sum": prefix_tokens_sum,
            "active_answer_tokens_sum": ans_tokens_sum,
            "masked_percentage": masked_pct,
            "active_percentage": active_pct
        }
    }

    stats_path = os.path.join(OUTPUT_DIR, "rag_sft_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved dataset statistics to {stats_path}")

    print("\nDataset generation and validation complete! All checks PASSED.")

if __name__ == "__main__":
    main()
