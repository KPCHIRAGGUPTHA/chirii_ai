import pytest
from data_cleaner import DataCleaner

def test_data_cleaner_filtering():
    cleaner = DataCleaner(min_doc_length=30, min_words=5, normalize_whitespace=True)

    # 1. Non-string document
    assert cleaner.clean_document(12345) is None
    assert cleaner.clean_document(None) is None

    # 2. Empty / whitespace document
    assert cleaner.clean_document("") is None
    assert cleaner.clean_document("   \n\t  ") is None

    # 3. Extremely short document
    assert cleaner.clean_document("Too short") is None

    # 4. Valid document
    valid_doc = "This is a valid document with sufficient length and words for pretraining."
    cleaned = cleaner.clean_document(valid_doc)
    assert cleaned is not None
    assert cleaned == valid_doc

    # 5. Whitespace normalization
    raw_text = "Line 1\r\n\r\n\r\n\r\nLine 2     with excessive    spaces."
    cleaned_norm = cleaner.clean_document(raw_text)
    assert "Line 1\n\nLine 2" in cleaned_norm
    assert "    " not in cleaned_norm

def test_data_cleaner_statistics():
    cleaner = DataCleaner(min_doc_length=20, min_words=3)
    cleaner.clean_document("Valid document text for testing.")
    cleaner.clean_document("")
    cleaner.clean_document(None)
    cleaner.clean_document("Short")

    stats = cleaner.get_statistics()
    assert stats["docs_processed"] == 4
    assert stats["docs_retained"] == 1
    assert stats["docs_removed_empty"] == 1
    assert stats["docs_removed_nonstring"] == 1
    assert stats["docs_removed_short"] == 1
    assert stats["doc_retention_rate_pct"] == 25.0
