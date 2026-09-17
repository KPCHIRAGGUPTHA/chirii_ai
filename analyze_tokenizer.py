import os
import json
from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer
from dataset import get_or_download_text

def run_tokenizer_analysis(output_dir: str = "results/phase4"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("         MINI-GPT TOKENIZER COMPRESSION & SUBWORD ANALYSIS          ")
    print("=" * 70)
    
    # 1. Load Tiny Shakespeare text
    text = get_or_download_text()
    
    # 2. Split train/val (90% train, 10% val)
    n = int(0.9 * len(text))
    train_text = text[:n]
    val_text = text[n:]
    
    print(f"Total Corpus Length:      {len(text):,} characters")
    print(f"Train Split Length:       {len(train_text):,} characters (90%)")
    print(f"Validation Split Length:  {len(val_text):,} characters (10%)")
    
    # 3. Train Tokenizers on TRAIN SPLIT ONLY
    print("\nTraining CharTokenizer on TRAIN split...")
    char_tok = CharTokenizer.from_text(train_text)
    
    print("Training BPETokenizer on TRAIN split (target_vocab_size=256)...")
    bpe_tok = BPETokenizer.train(train_text, target_vocab_size=256)
    
    # 4. Compare Tokenization Metrics over Validation Set
    val_char_tokens = char_tok.encode(val_text)
    val_bpe_tokens = bpe_tok.encode(val_text)
    
    val_char_count = len(val_text)
    val_char_token_count = len(val_char_tokens)
    val_bpe_token_count = len(val_bpe_tokens)
    
    char_avg_chars_per_token = val_char_count / val_char_token_count
    bpe_avg_chars_per_token = val_char_count / val_bpe_token_count
    
    token_count_reduction_percent = round(((val_char_token_count - val_bpe_token_count) / val_char_token_count) * 100.0, 2)
    bpe_compression_gain_pct = round(((bpe_avg_chars_per_token - char_avg_chars_per_token) / char_avg_chars_per_token) * 100.0, 2)
    
    # 5. Analyze Sample Texts
    sample_texts = [
        "First Citizen:",
        "To be, or not to be, that is the question:",
        "ROMEO:\nSoft! what light through yonder window breaks?",
        "JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?",
        "HAMLET:\nSpeak the speech, I pray you, as I pronounced it to you."
    ]
    
    samples_analysis = []
    for sample in sample_texts:
        c_tokens = char_tok.encode(sample)
        b_tokens = bpe_tok.encode(sample)
        
        # Subword representation for BPE
        b_subwords = [bpe_tok.itos.get(tid, '<unk>') for tid in b_tokens]
        
        samples_analysis.append({
            "text": sample,
            "char_count": len(sample),
            "char_token_count": len(c_tokens),
            "bpe_token_count": len(b_tokens),
            "avg_chars_per_token": round(len(sample) / len(b_tokens), 2),
            "compression_ratio": round(len(sample) / len(b_tokens), 2),
            "bpe_tokens": b_tokens,
            "bpe_subwords": b_subwords
        })
        
    analysis_data = {
        "dataset_info": {
            "total_chars": len(text),
            "train_chars": len(train_text),
            "val_chars": len(val_text)
        },
        "char_tokenizer": {
            "vocab_size": char_tok.vocab_size,
            "val_token_count": val_char_token_count,
            "avg_chars_per_token": round(char_avg_chars_per_token, 4),
            "compression_ratio": round(char_avg_chars_per_token, 4),
            "avg_tokens_per_char": round(val_char_token_count / val_char_count, 4)
        },
        "bpe_tokenizer": {
            "vocab_size": bpe_tok.vocab_size,
            "num_merges": len(bpe_tok.merges),
            "val_token_count": val_bpe_token_count,
            "avg_chars_per_token": round(bpe_avg_chars_per_token, 4),
            "token_count_reduction_percent": token_count_reduction_percent,
            "compression_ratio": round(bpe_avg_chars_per_token, 4),
            "avg_tokens_per_char": round(val_bpe_token_count / val_char_count, 4),
            "compression_improvement_percent": bpe_compression_gain_pct
        },
        "samples": samples_analysis
    }
    
    # Write JSON analysis
    json_path = os.path.join(output_dir, "tokenizer_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    print(f"\nWrote tokenizer analysis JSON to {json_path}")
    
    # Write Markdown analysis report
    md_path = os.path.join(output_dir, "tokenizer_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Tokenizer Analysis: CharTokenizer vs BPETokenizer\n\n")
        f.write("## 1. Summary Comparison\n\n")
        f.write("| Metric | CharTokenizer | BPETokenizer | Difference / Improvement |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Vocabulary Size** | `{char_tok.vocab_size}` | `{bpe_tok.vocab_size}` | +{bpe_tok.vocab_size - char_tok.vocab_size} tokens |\n")
        f.write(f"| **Validation Token Count** | `{val_char_token_count:,}` | `{val_bpe_token_count:,}` | -{val_char_token_count - val_bpe_token_count:,} tokens ({token_count_reduction_percent}% reduction) |\n")
        f.write(f"| **Average Characters per Token** | `{char_avg_chars_per_token:.4f}` chars/tok | `{bpe_avg_chars_per_token:.4f}` chars/tok | **+{bpe_compression_gain_pct}% higher** |\n")
        f.write(f"| **Avg Tokens per Char** | `{val_char_token_count / val_char_count:.4f}` | `{val_bpe_token_count / val_char_count:.4f}` | {((val_bpe_token_count / val_char_count) - 1.0)*100:.2f}% |\n\n")
        
        f.write("## 2. Sample Tokenizations\n\n")
        for i, sample in enumerate(samples_analysis, 1):
            f.write(f"### Sample {i}\n\n")
            f.write(f"**Original Text** ({sample['char_count']} chars):\n```text\n{sample['text']}\n```\n\n")
            f.write(f"- **Char Token Count**: `{sample['char_token_count']}` tokens\n")
            f.write(f"- **BPE Token Count**: `{sample['bpe_token_count']}` tokens (Average Chars/Token: `{sample['avg_chars_per_token']}`)\n")
            f.write(f"- **BPE Subword Decomposition**: `{sample['bpe_subwords']}`\n\n")
            
    print(f"Wrote tokenizer analysis Markdown to {md_path}")
    return bpe_tok, analysis_data

if __name__ == "__main__":
    run_tokenizer_analysis()
