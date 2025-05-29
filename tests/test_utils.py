# tests/test_utils.py
import pytest
import sys
import os
from nltk.tokenize import word_tokenize # Import for smart_truncate_text tests

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

# --- Tests for count_words (assuming it now uses text.split()) --- #

def test_count_words_empty():
    assert utils.count_words("") == 0

def test_count_words_simple():
    assert utils.count_words("Hello world") == 2

def test_count_words_with_punctuation():
    # text.split() behavior: "Hello, world!".split() -> ['Hello,', 'world!'] (2 words)
    assert utils.count_words("Hello, world!") == 2 

def test_count_words_with_newlines():
    # text.split() behavior: "First line.\nSecond line.".split() -> ['First', 'line.', 'Second', 'line.'] (4 words)
    assert utils.count_words("First line.\nSecond line.") == 4 

# --- Tests for check_word_count_adherence --- #

@pytest.mark.parametrize("actual, target, tolerance, expected_adherent, expected_dev_approx", [
    # Existing tests for explicit tolerances
    (100, 100, 0.20, True, 0.0),
    (80, 100, 0.20, True, -0.20),
    (120, 100, 0.20, True, 0.20),
    (79, 100, 0.20, False, -0.21),
    (121, 100, 0.20, False, 0.21),
    (50, 50, 0.10, True, 0.0),
    (44, 50, 0.10, False, -0.12), # 50 * 0.9 = 45
    (56, 50, 0.10, False, 0.12),  # 50 * 1.1 = 55
    (0, 0, 0.20, True, 0.0),    
    (10, 0, 0.20, False, 0.0),   
    # New tests for the 0.50 default tolerance (tolerance parameter will be None in call)
    (100, 100, None, True, 0.0),      # Exact match
    (50, 100, None, True, -0.50),     # Lower bound (100 * (1-0.5) = 50)
    (150, 100, None, True, 0.50),    # Upper bound (100 * (1+0.5) = 150)
    (49, 100, None, False, -0.51),   # Just outside lower bound
    (151, 100, None, False, 0.51),   # Just outside upper bound
    (75, 100, None, True, -0.25),    # Well within 50% tolerance
    (125, 100, None, True, 0.25),    # Well within 50% tolerance
])
def test_check_word_count_adherence(actual, target, tolerance, expected_adherent, expected_dev_approx):
    if tolerance is None: # Test default tolerance
        is_adherent, deviation = utils.check_word_count_adherence(actual, target)
    else: # Test with explicit tolerance
        is_adherent, deviation = utils.check_word_count_adherence(actual, target, tolerance)
    assert is_adherent == expected_adherent
    assert abs(deviation - expected_dev_approx) < 0.001 

# --- Tests for smart_truncate_text --- #
# Assuming smart_truncate_text internally still uses nltk.word_tokenize for its logic

TEXT_FOR_TRUNCATION = "This is the first sentence. This is the second sentence, which is a bit longer. And finally, the third sentence is here to make it long enough for truncation exercises."
# NLTK word_tokenize(TEXT_FOR_TRUNCATION) -> 36 tokens

def test_smart_truncate_already_short():
    text = "This is short enough."
    # NLTK tokens: 5
    assert utils.smart_truncate_text(text, 10, max_overshoot_words=2) == text

def test_smart_truncate_slightly_over_within_overshoot():
    text = "This is just a tiny bit over the target allowed."
    # NLTK tokens: 11
    assert utils.smart_truncate_text(text, 9, max_overshoot_words=2) == text

def test_smart_truncate_needs_truncation_simple():
    target_wc_nltk = 5 # Target based on NLTK tokens for smart_truncate_text internal logic
    original_text = "This is the first sentence."
    # NLTK tokens of original_text: ['This', 'is', 'the', 'first', 'sentence', '.'] -> 6 tokens
    truncated = utils.smart_truncate_text(original_text, target_wc_nltk, max_overshoot_words=0)
    # smart_truncate_text truncates to target_wc_nltk NLTK tokens
    assert len(word_tokenize(truncated)) == target_wc_nltk
    # Expected based on NLTK word_tokenize: words[:5] -> ['This', 'is', 'the', 'first', 'sentence']
    # Joined: "This is the first sentence"
    assert truncated == "This is the first sentence"

def test_smart_truncate_longer_text():
    target_wc_nltk = 15 
    # TEXT_FOR_TRUNCATION has 36 NLTK tokens
    truncated = utils.smart_truncate_text(TEXT_FOR_TRUNCATION, target_wc_nltk, max_overshoot_words=3)
    # Assert based on NLTK token count for the result, as smart_truncate_text uses it
    assert len(word_tokenize(truncated)) == target_wc_nltk
    expected_start = "This is the first sentence . This is the second sentence , which is a"
    assert truncated == expected_start

# It would be good to also test the old filename utils if they are still used.
# For now, assuming they are being phased out by the new exporter logic.

# To run these tests:
# 1. Make sure pytest is installed (pip install pytest)
# 2. Navigate to your project root directory in the terminal (where tests/ folder is)
# 3. Run the command: pytest 