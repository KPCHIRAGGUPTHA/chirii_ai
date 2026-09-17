# Phase 4 — BPE Tokenizer

## 1. Objective

The objective of Phase 4 is to replace the character-level tokenizer with a Byte Pair Encoding (BPE) subword tokenizer implemented from scratch. This controlled experiment evaluates how subword tokenization affects language model capacity, sequence compression, context window efficiency, parameter count, and loss/perplexity/BPC metrics.

---

## 2. Character Tokenization

The baseline `CharTokenizer` maps individual characters directly to integer IDs:
- **Vocabulary Size**: `66` unique characters (letters, numbers, punctuation, spaces, newlines).
- **Sequence Length**: 1 character = 1 token.
- **Limitation**: The model must expend significant capacity learning basic spelling before learning higher-level word semantics or syntactic relationships.

---

## 3. What is BPE?

Byte Pair Encoding (BPE) is a data-driven subword tokenization algorithm. It begins with base characters and iteratively merges the most frequently co-occurring adjacent token pairs into new subword units (e.g. `"t" + "h" -> "th"`, `"t" + "o" -> "to "`). This creates a hybrid vocabulary containing single characters, frequent subword chunks, and common full words.

---

## 4. BPE Training Algorithm

1. **Base Vocabulary**: Extract unique characters from the training text corpus.
2. **Frequency Counting**: Count frequencies of all adjacent token pairs `(token_a, token_b)`.
3. **Merge Selection**: Identify the pair with highest frequency (using deterministic lexicographical tie-breaking for equal counts).
4. **Merge Execution**: Substitute all co-occurrences of `token_a + token_b` with a new merged token string.
5. **Iteration**: Repeat until the target vocabulary size (`vocab_size = 256`) is reached.

*Training Isolation*: The BPE vocabulary and merge rules were trained **strictly on the 90% training split** (`train_text`) of Tiny Shakespeare to prevent validation split data leakage. `val_text` was never accessed during tokenizer training.

---

## 5. Tokenization Examples

| Text Sample | Char Token Count | BPE Token Count | Average Characters per Token | BPE Subword Decomposition |
| :--- | :---: | :---: | :---: | :--- |
| `"First Citizen:"` | 14 | 9 | **1.56** | `['F', 'ir', 'st ', 'C', 'it', 'i', 'z', 'en', ':']` |
| `"To be, or not to be, that is the question:"` | 42 | 18 | **2.33** | `['To ', 'b', 'e, ', 'or', ' ', 'not ', 'to ', 'b', 'e, ', 'that ', 'is ', 'the ', 'q', 'u', 'es', 't', 'ion', ':']` |
| `"ROMEO:\\nSoft! what light..."` | 53 | 32 | **1.66** | `['R', 'O', 'M', 'E', 'O:\\n', 'S', 'of', 't', '!', ' w', 'hat ', ...]` |

---

## 6. Compression & Reduction Comparison

- **CharTokenizer**: **1.0000** average characters / token
- **BPETokenizer**: **1.8342** average characters / token
- **Token Count Reduction (%)**: **45.48%** token count reduction over validation set

---

## 7. Vocabulary Comparison

- **CharTokenizer Vocab Size**: `66` tokens
- **BPETokenizer Vocab Size**: `256` tokens (`66` base characters + `190` learned BPE merges)

---

## 8. Parameter Count Impact

All token embedding tables (`wte`) and language model heads (`lm_head`) scale linearly with vocabulary size ($V \times E$):
- **CharTokenizer Model (`vocab_size=66`)**: **818,176** parameters
- **BPETokenizer Model (`vocab_size=256`)**: **842,496** parameters (+24,320 parameters, or **+2.97%**)

---

## 9. Training & Validation Results

| Experiment | Tokenizer | Vocab Size | Parameters | Avg Chars/Token | Token Reduction (%) | Train Loss | Best Val Loss | Best Val PPL | Best Val BPC | Eval Loss | Eval PPL | Eval BPC | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`char_baseline`** | `CharTokenizer` | `66` | 818,176 | 1.00 | 0.00% | 2.0241 | 2.0936 | 8.1142 | 3.0205 | 2.0898 | 8.0832 | 3.0149 | 569.6s |
| **`bpe`** | `BPETokenizer` | `256` | 842,496 | **1.83** | **45.48%** | 3.2324 | 3.4769 | 32.3590 | 2.7347 | 3.4924 | 32.8646 | 2.7469 | 524.4s |

---

## 10. Perplexity & Bits Per Character (BPC) Interpretation

### Cross-Tokenizer Metrics
- **Perplexity (PPL)**: Represents $\exp(\mathcal{L}_{\text{token}})$. Because BPE subword tokens span multiple characters, BPE per-token perplexity measures uncertainty across 256 subword choices rather than 66 character choices.
- **Bits Per Character (BPC)**: Defined as $\text{BPC} = \frac{\mathcal{L}_{\text{val}} \cdot N_{\text{val\_tokens}}}{\ln(2) \cdot N_{\text{val\_chars}}}$. BPC standardizes cross-entropy loss to bits per character, enabling direct, fair comparison across different tokenizers.

---

## 11. Text Generation Comparison

### Prompt: `ROMEO:`

- **`char_baseline`**:
  ```text
  ROMEO:
  Hat our thee gody speak, would the father lear
  That woman to to me my face thee
  ```

- **`bpe`**:
  ```text
  ROMEO:
  Good sir, I have serve user and look:
  I'll prove standard for your grace.
  ```

*Observation*: Qualitative inspection suggests that the BPE model produced more recognizable word and subword structures in the observed samples. This observation is qualitative and does not establish statistical significance.

---

## 12. What We Learned

1. **Sequence Length Reduction**: BPE compresses sequence length significantly (**~45.5%** token count reduction).
2. **Context Window Amplification**: A fixed Transformer context window (`block_size=128`) spans nearly double the text when backed by subwords (~235 characters vs 128 characters).
3. **Cross-Tokenizer Metric Standardization**: BPC provides a comparable metric across character and subword tokenizations.

---

## 13. Limitations

- Small BPE vocabulary size (`vocab_size = 256`).
- Fixed 600 iteration training budget on CPU.
- Corpus limited to Tiny Shakespeare.

---

## 14. Phase 5 Recommendations

In Phase 5, transition to a large-scale pretraining corpus (FineWeb) and scale BPE vocabulary size to handle multi-domain language modeling effectively.
