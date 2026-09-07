import tempfile
import os
from tokenizer import CharTokenizer

def test_vocabulary_size():
    tokenizer = CharTokenizer()
    assert tokenizer.vocab_size == 97, f"Expected vocabulary size 97, got {tokenizer.vocab_size}"

def test_special_characters():
    tokenizer = CharTokenizer()
    encoded = tokenizer.encode("Hello \u1234")
    assert encoded[-1] == tokenizer.unk_id, "Out-of-vocab character should encode to unk_id"

def test_roundtrip():
    tokenizer = CharTokenizer()
    text = "Hello, Mini-GPT World!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text, "Tokenizer round-trip failed"

def test_from_text():
    text = "abc standard"
    tokenizer = CharTokenizer.from_text(text)
    assert 'a' in tokenizer.stoi
    assert 'b' in tokenizer.stoi
    assert 'c' in tokenizer.stoi
    assert '<unk>' in tokenizer.stoi

def test_save_load_roundtrip():
    text = "Quick brown fox"
    tok1 = CharTokenizer.from_text(text)
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.json")
        tok1.save(vocab_path)
        tok2 = CharTokenizer.load(vocab_path)
        assert tok1.vocab_size == tok2.vocab_size
        assert tok1.stoi == tok2.stoi
        assert tok1.itos == tok2.itos
        assert tok1.unk_id == tok2.unk_id

def test_empty_string():
    tokenizer = CharTokenizer()
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""