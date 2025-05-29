# tests/test_utils.py
import pytest
import sys
import os

# Add the src directory to the Python path to allow imports of utils
# This is a common pattern for structuring tests.
# Adjust if your project structure or test runner handles this differently.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "src")
sys.path.insert(0, src_dir)

import utils # Now we can import from src.utils

# --- Tests for clean_generated_text --- #

def test_clean_generated_text_empty():
    assert utils.clean_generated_text("") == ""

def test_clean_generated_text_no_change():
    assert utils.clean_generated_text("This is clean.") == "This is clean."

def test_clean_generated_text_strip_whitespace():
    assert utils.clean_generated_text("  leading and trailing  ") == "leading and trailing"

def test_clean_generated_text_only_whitespace():
    assert utils.clean_generated_text("   \n\t  ") == ""

# --- Tests for count_words --- #

def test_count_words_empty():
    assert utils.count_words("") == 0

def test_count_words_simple():
    assert utils.count_words("Hello world") == 2

def test_count_words_with_punctuation():
    # nltk.word_tokenize typically counts punctuation as separate tokens if not handled
    # Depending on desired behavior, this test might need adjustment or the function itself.
    # Current nltk.word_tokenize behavior: ['Hello', ',', 'world', '!'] -> 4
    # If we want to count only "words", this test would fail and function might need refinement.
    # For now, assume we accept NLTK's default tokenization for word count.
    assert utils.count_words("Hello, world!") == 4 

def test_count_words_with_newlines():
    assert utils.count_words("First line.\nSecond line.") == 6 # ['First', 'line', '.', 'Second', 'line', '.']

# --- Tests for check_word_count_adherence --- #

@pytest.mark.parametrize("actual, target, tolerance, expected_adherent, expected_dev_approx", [
    (100, 100, 0.20, True, 0.0),
    (80, 100, 0.20, True, -0.20),
    (120, 100, 0.20, True, 0.20),
    (79, 100, 0.20, False, -0.21),
    (121, 100, 0.20, False, 0.21),
    (50, 50, 0.10, True, 0.0),
    (44, 50, 0.10, False, -0.12), # 50 * 0.9 = 45
    (56, 50, 0.10, False, 0.12),  # 50 * 1.1 = 55
    (0, 0, 0.20, True, 0.0),    # Edge case: zero target, zero actual
    (10, 0, 0.20, False, 0.0),   # Edge case: zero target, non-zero actual (dev is tricky, func returns 0)
])
def test_check_word_count_adherence(actual, target, tolerance, expected_adherent, expected_dev_approx):
    is_adherent, deviation = utils.check_word_count_adherence(actual, target, tolerance)
    assert is_adherent == expected_adherent
    # For deviation, allow for small floating point inaccuracies
    assert abs(deviation - expected_dev_approx) < 0.001 

# --- Tests for smart_truncate_text --- #

TEXT_FOR_TRUNCATION = "This is the first sentence. This is the second sentence, which is a bit longer. And finally, the third sentence is here to make it long enough for truncation exercises."
# Word count (NLTK default): ['This', 'is', 'the', 'first', 'sentence', '.', 'This', 'is', 'the', 'second', 'sentence', ',', 'which', 'is', 'a', 'bit', 'longer', '.', 'And', 'finally', ',', 'the', 'third', 'sentence', 'is', 'here', 'to', 'make', 'it', 'long', 'enough', 'for', 'truncation', 'exercises', '.'] -> 36 words by NLTK count_words

def test_smart_truncate_already_short():
    text = "This is short enough."
    # NLTK words: ['This', 'is', 'short', 'enough', '.'] -> 5
    assert utils.smart_truncate_text(text, 10, max_overshoot_words=2) == text

def test_smart_truncate_slightly_over_within_overshoot():
    text = "This is just a tiny bit over the target allowed."
    # NLTK words: ['This', 'is', 'just', 'a', 'tiny', 'bit', 'over', 'the', 'target', 'allowed', '.'] -> 11
    assert utils.smart_truncate_text(text, 9, max_overshoot_words=2) == text

def test_smart_truncate_needs_truncation_simple():
    target_wc = 5
    # Original: "This is the first sentence." (6 NLTK words)
    # Expected: "This is the first sentence" (words[:5] -> ['This', 'is', 'the', 'first', 'sentence'])
    truncated = utils.smart_truncate_text("This is the first sentence.", target_wc, max_overshoot_words=0)
    assert utils.count_words(truncated) == target_wc
    assert truncated == "This is the first sentence"

def test_smart_truncate_longer_text():
    target_wc = 15 
    # TEXT_FOR_TRUNCATION has 36 NLTK words
    truncated = utils.smart_truncate_text(TEXT_FOR_TRUNCATION, target_wc, max_overshoot_words=3)
    assert utils.count_words(truncated) == target_wc
    # The exact output depends on NLTK tokenization and the simple join. 
    # For words[:15] of TEXT_FOR_TRUNCATION, it is:
    # ['This', 'is', 'the', 'first', 'sentence', '.', 'This', 'is', 'the', 'second', 'sentence', ',', 'which', 'is', 'a']
    # Joined: "This is the first sentence . This is the second sentence , which is a"
    # This highlights a limitation of simple space-joining after tokenization. 
    # A more robust test would check word count and perhaps if it starts the same way.
    # For now, we focus on word count primarily for this test after truncation.
    expected_start = "This is the first sentence . This is the second sentence , which is a"
    assert truncated == expected_start

# It would be good to also test the old filename utils if they are still used.
# For now, assuming they are being phased out by the new exporter logic.

# To run these tests:
# 1. Make sure pytest is installed (pip install pytest)
# 2. Navigate to your project root directory in the terminal (where tests/ folder is)
# 3. Run the command: pytest 