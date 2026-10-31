"""Phase 10A tests: tokenizer v2 and text pinning. Tokenizer tests use a small trained tokenizer, and the real one if it exists."""

import hashlib
import os
import tempfile
import unittest

from tokenizers import Tokenizer

from phase10.common import SPECIAL_TOKENS, TOKENIZER_PATH, project_path, read_text_lf, sha256_file, sha256_text
from phase10.collect_data import split_for_document
from phase10.tokenizer_v2 import VOCAB_SIZE, decode, encode, token_byte_lengths, train_tokenizer

SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces, trailing spaces  ",
    "line one\nline two\r\nline three\tTabbed\n\n",
    "Unicode: café, naïve, Zürich, 日本語のテキスト, Ελληνικά, русский, العربية, 🙂🚀 and 👩‍💻.",
    "def f(x):\n    return {'a': x ** 2, \"b\": [1, 2, 3]}  # comment\n\nif __name__ == '__main__':\n\tprint(f(10))\n",
    "#include <stdio.h>\nint main(void) { printf(\"%d\\n\", 42 + 7 * (3 - 1)); return 0; }",
    "Numbers: 0 1 23 456.789 1,000,000 -3.14e-10 2026-10-31 $36.50 50% 007",
    "Other number forms: ٣٤ ½ ² Ⅷ",
    "Control-ish: \x00\x01\x7f and a non-breaking space ​ zero width",
    "<|user|>What is 2+2?<|assistant|><|calc|>2+2<|/calc|><|result|>4<|/result|>The answer is 4.<|endoftext|>",
    "<|notspecial|> <| <|endoftext <user> |>",
]


def tiny_corpus():
    for _ in range(30):
        yield from SAMPLES


class TokenizerBehaviour:
    """Runs against any trained tokenizer, set as self.tokenizer by the subclasses below."""

    def test_round_trip_returns_identical_text(self):
        for text in SAMPLES:
            self.assertEqual(decode(self.tokenizer, encode(self.tokenizer, text)), text)

    def test_every_digit_is_a_single_token(self):
        for digit in "0123456789":
            self.assertIsNotNone(self.tokenizer.token_to_id(digit))
            self.assertEqual(encode(self.tokenizer, digit), [self.tokenizer.token_to_id(digit)])

    def test_digits_in_numbers_are_split_individually(self):
        digit_ids = {self.tokenizer.token_to_id(d): d for d in "0123456789"}
        for text, digits in (("12345", "12345"), ("in 2026, 1,000 and x9y", "20261000" + "9"), (" 7", "7"), ("3.14", "314")):
            ids = encode(self.tokenizer, text)
            self.assertEqual("".join(digit_ids[i] for i in ids if i in digit_ids), digits, text)

    def test_no_merged_token_contains_a_digit(self):
        for token in self.tokenizer.get_vocab():
            if token not in SPECIAL_TOKENS and any(c in "0123456789" for c in token):
                self.assertEqual(len(token), 1, token)

    def test_special_tokens_are_never_split(self):
        for token in SPECIAL_TOKENS:
            token_id = self.tokenizer.token_to_id(token)
            self.assertIsNotNone(token_id)
            self.assertEqual(encode(self.tokenizer, token), [token_id])
            for text in (f"abc{token}def", f"{token}{token}", f" {token} ", f"12{token}34"):
                ids = encode(self.tokenizer, text)
                self.assertEqual(ids.count(token_id), text.count(token), text)
                self.assertEqual(decode(self.tokenizer, ids), text)

    def test_special_tokens_have_the_first_ids(self):
        self.assertEqual([self.tokenizer.token_to_id(t) for t in SPECIAL_TOKENS], list(range(len(SPECIAL_TOKENS))))

    def test_token_byte_lengths_add_up_to_the_utf8_size(self):
        lengths = token_byte_lengths(self.tokenizer)
        for text in SAMPLES:
            if not any(t in text for t in SPECIAL_TOKENS):
                self.assertEqual(int(lengths[encode(self.tokenizer, text)].sum()), len(text.encode("utf-8")), text)

    def test_special_tokens_count_as_zero_bytes(self):
        lengths = token_byte_lengths(self.tokenizer)
        for token in SPECIAL_TOKENS:
            self.assertEqual(int(lengths[self.tokenizer.token_to_id(token)]), 0)


class TinyTokenizerTest(TokenizerBehaviour, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = train_tokenizer(tiny_corpus(), vocab_size=600)


@unittest.skipUnless(os.path.exists(project_path(TOKENIZER_PATH)), "Real tokenizer not trained yet")
class RealTokenizerTest(TokenizerBehaviour, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))

    def test_vocabulary_size(self):
        self.assertEqual(self.tokenizer.get_vocab_size(), VOCAB_SIZE)


class TextPinTest(unittest.TestCase):
    def write(self, data):
        handle, path = tempfile.mkstemp()
        os.write(handle, data)
        os.close(handle)
        self.addCleanup(os.remove, path)
        return path

    def test_crlf_and_lf_files_get_the_same_pin(self):
        lf = self.write(b"one\ntwo\n\nthree")
        crlf = self.write(b"one\r\ntwo\r\n\r\nthree")
        self.assertNotEqual(sha256_file(lf), sha256_file(crlf))
        self.assertEqual(sha256_text(lf), sha256_text(crlf))

    def test_lf_pin_equals_the_plain_hash_of_lf_text(self):
        path = self.write(b"one\ntwo\n")
        self.assertEqual(sha256_text(path), sha256_file(path))

    def test_crlf_split_across_read_chunks_is_normalised(self):
        lf = self.write(b"abc\ndef\n")
        crlf = self.write(b"abc\r\ndef\r\n")
        for chunk_size in range(1, 12):
            self.assertEqual(sha256_text(crlf, chunk_size=chunk_size), sha256_text(lf), chunk_size)

    def test_lone_cr_is_kept(self):
        for chunk_size in (1, 2, 3, 1024):
            self.assertNotEqual(sha256_text(self.write(b"a\rb\n"), chunk_size=chunk_size), sha256_text(self.write(b"a\nb\n")))
            self.assertNotEqual(sha256_text(self.write(b"ab\r"), chunk_size=chunk_size), sha256_text(self.write(b"ab")))
        self.assertEqual(sha256_text(self.write(b"\r\r\n"), chunk_size=1), hashlib.sha256(b"\r\n").hexdigest())

    def test_changed_content_changes_the_pin(self):
        self.assertNotEqual(sha256_text(self.write(b"one\r\n")), sha256_text(self.write(b"two\r\n")))

    def test_read_text_lf_matches_its_pin(self):
        path = self.write("é\r\nb\r\n".encode("utf-8"))
        self.assertEqual(hashlib.sha256(read_text_lf(path).encode("utf-8")).hexdigest(), sha256_text(path))

    def test_old_validation_text_pin_is_the_git_blob(self):
        path = project_path("data/phase5d/val_corpus.txt")
        if os.path.exists(path):
            self.assertEqual(sha256_text(path), "6c5c425a42b6fa988afc6a081c9bb01773b449308165934c5bc3038789c6e683")


class SplitTest(unittest.TestCase):
    def test_split_is_deterministic_and_near_half_a_percent(self):
        docs = [f"document {i}".encode() for i in range(200_000)]
        splits = [split_for_document(d) for d in docs]
        self.assertEqual(splits, [split_for_document(d) for d in docs])
        self.assertAlmostEqual(splits.count("val") / len(splits), 0.005, delta=0.001)


if __name__ == "__main__":
    unittest.main()
