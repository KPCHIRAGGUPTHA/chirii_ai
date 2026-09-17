# Tokenizer Analysis: CharTokenizer vs BPETokenizer

## 1. Summary Comparison

| Metric | CharTokenizer | BPETokenizer | Difference / Improvement |
| :--- | :---: | :---: | :---: |
| **Vocabulary Size** | `66` | `256` | +190 tokens |
| **Validation Token Count** | `111,540` | `60,811` | -50,729 tokens (45.48% reduction) |
| **Compression Ratio** | `1.0000` chars/tok | `1.8342` chars/tok | **+83.42% higher compression** |
| **Avg Tokens per Char** | `1.0000` | `0.5452` | -45.48% |

## 2. Sample Tokenizations

### Sample 1

**Original Text** (14 chars):
```text
First Citizen:
```

- **Char Token Count**: `14` tokens
- **BPE Token Count**: `9` tokens (Compression: `1.56` chars/token)
- **BPE Subword Decomposition**: `['F', 'ir', 'st ', 'C', 'it', 'i', 'z', 'en', ':']`

### Sample 2

**Original Text** (42 chars):
```text
To be, or not to be, that is the question:
```

- **Char Token Count**: `42` tokens
- **BPE Token Count**: `18` tokens (Compression: `2.33` chars/token)
- **BPE Subword Decomposition**: `['To ', 'b', 'e, ', 'or', ' ', 'not ', 'to ', 'b', 'e, ', 'that ', 'is ', 'the ', 'q', 'u', 'es', 't', 'ion', ':']`

### Sample 3

**Original Text** (53 chars):
```text
ROMEO:
Soft! what light through yonder window breaks?
```

- **Char Token Count**: `53` tokens
- **BPE Token Count**: `32` tokens (Compression: `1.66` chars/token)
- **BPE Subword Decomposition**: `['R', 'O', 'M', 'E', 'O:\n', 'S', 'of', 't', '!', ' w', 'hat ', 'li', 'gh', 't ', 'th', 'r', 'ou', 'gh', ' ', 'y', 'on', 'd', 'er', ' w', 'in', 'd', 'ow', ' b', 'rea', 'k', 's', '?']`

### Sample 4

**Original Text** (49 chars):
```text
JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
```

- **Char Token Count**: `49` tokens
- **BPE Token Count**: `34` tokens (Compression: `1.44` chars/token)
- **BPE Subword Decomposition**: `['J', 'U', 'L', 'I', 'E', 'T', ':\n', 'O', ' ', 'R', 'om', 'e', 'o', ', ', 'R', 'om', 'e', 'o', '!', ' w', 'h', 'er', 'e', 'for', 'e ', 'ar', 't ', 'thou', ' ', 'R', 'om', 'e', 'o', '?']`

### Sample 5

**Original Text** (64 chars):
```text
HAMLET:
Speak the speech, I pray you, as I pronounced it to you.
```

- **Char Token Count**: `64` tokens
- **BPE Token Count**: `35` tokens (Compression: `1.83` chars/token)
- **BPE Subword Decomposition**: `['H', 'A', 'M', 'L', 'E', 'T', ':\n', 'S', 'p', 'ea', 'k', ' the ', 's', 'p', 'ee', 'ch', ', ', 'I ', 'p', 'ra', 'y ', 'you', ', ', 'as ', 'I ', 'p', 'r', 'on', 'oun', 'c', 'ed ', 'it ', 'to ', 'you', '.']`

