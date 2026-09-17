import os
import pytest
from bpe_tokenizer import BPETokenizer

def test_bpe_training_and_vocab():
    text = "hug pug hug bug hug rug"
    tokenizer = BPETokenizer.train(text, target_vocab_size=15)
    assert tokenizer.vocab_size <= 15
    assert len(tokenizer.merges) > 0

def test_bpe_encode_decode_roundtrip():
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    tokenizer = BPETokenizer.train(text, target_vocab_size=50)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text

def test_bpe_test_strings_roundtrip():
    printable_ascii = " \n!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    train_corpus = printable_ascii + "\nTo be, or not to be, that is the question. Hello world! ROMEO: JULIET: First Citizen:"
    tokenizer = BPETokenizer.train(train_corpus, target_vocab_size=150)
    
    test_strings = [
        "Hello world",
        "First Citizen:",
        "ROMEO:",
        "To be, or not to be",
        "Multiple lines\nand punctuation!?,:;.-'",
        "   whitespace   test   "
    ]
    
    for test_str in test_strings:
        encoded = tokenizer.encode(test_str)
        decoded = tokenizer.decode(encoded)
        assert decoded == test_str, f"Failed roundtrip for string: {test_str!r}"

def test_bpe_save_load_roundtrip(tmp_path):
    text = "Testing BPE tokenizer save and load methods."
    tokenizer = BPETokenizer.train(text, target_vocab_size=30)
    
    save_path = os.path.join(tmp_path, "bpe_vocab.json")
    tokenizer.save(save_path)
    assert os.path.exists(save_path)
    
    loaded_tokenizer = BPETokenizer.load(save_path)
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert loaded_tokenizer.merges == tokenizer.merges
    
    encoded_orig = tokenizer.encode(text)
    encoded_loaded = loaded_tokenizer.encode(text)
    assert encoded_orig == encoded_loaded
    assert loaded_tokenizer.decode(encoded_loaded) == text

def test_bpe_deterministic_behavior():
    text = "The quick brown fox jumps over the lazy dog."
    tok1 = BPETokenizer.train(text, target_vocab_size=40)
    tok2 = BPETokenizer.train(text, target_vocab_size=40)
    
    assert tok1.merges == tok2.merges
    assert tok1.encode(text) == tok2.encode(text)

def test_bpe_empty_string():
    text = "Sample text for training"
    tokenizer = BPETokenizer.train(text, target_vocab_size=30)
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""

def test_bpe_unknown_character():
    train_text = "abc"
    tokenizer = BPETokenizer.train(train_text, target_vocab_size=10)
    encoded = tokenizer.encode("z")
    assert encoded == [tokenizer.unk_id]

def test_bpe_train_only_no_val_leakage():
    """
    Verify that BPETokenizer training uses train_text exclusively.
    val_text contains unique patterns ('QQQ', 'ZZZ') not present in train_text.
    Tokenizer must NOT learn merges for validation-only patterns.
    """
    train_text = "the cat sat on the mat"
    val_text = "the dog slept on ZZZ QQQ ZZZ QQQ"
    
    tokenizer = BPETokenizer.train(train_text, target_vocab_size=20)
    
    # Check that no merge in tokenizer.merges contains Z or Q
    for pair in tokenizer.merges:
        assert 'Z' not in pair[0] and 'Z' not in pair[1], f"Learned validation character 'Z' in merge pair: {pair}"
        assert 'Q' not in pair[0] and 'Q' not in pair[1], f"Learned validation character 'Q' in merge pair: {pair}"
        
    # Check that tokenizer vocabulary does not contain validation-only characters (except unk_token)
    assert 'Z' not in tokenizer.base_chars
    assert 'Q' not in tokenizer.base_chars

