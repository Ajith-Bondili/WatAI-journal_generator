import datetime
import re
from nltk.tokenize import word_tokenize, sent_tokenize

# def get_current_datetime_str(format_str="%Y%m%d") -> str:
#     """Returns the current date as a string, formatted as YYYYMMDD."""
#     return datetime.datetime.now().strftime(format_str)

def get_current_datetime_str_for_file_id() -> str:
    """Returns the current datetime as a string suitable for a unique file ID: YYYYMMDD_HHMMSSffffff."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")

# generate_file_id is no longer used by the exporter, can be deprecated/removed if not used elsewhere
# def generate_file_id(entry_index: int, total_entries_for_day: int) -> str:
#     """Generates a unique ID for an entry within a day, e.g., 001, 002."""
#     padding = len(str(total_entries_for_day))
#     return f"{entry_index:0{padding}d}"

def construct_filename(unique_id_str: str, prefix: str = "journal") -> str:
    """Constructs the filename using a unique ID string, e.g., journal_YYYYMMDD_HHMMSSffffff.txt."""
    return f"{prefix}_{unique_id_str}.txt"

def clean_generated_text(text: str) -> str:
    """
    Basic cleaning of LLM generated text.
    - Removes text that might be part of the prompt/instruction included in the output.
    - Strips leading/trailing whitespace.
    """
    # Models sometimes repeat the prompt or instructions. 
    # This is a common pattern for some models if they include the input in the output.
    # For Sarvam-M, the response is typically in a structured format where generated_text is separate.
    # However, if the raw output of a text-generation pipeline for other models includes the prompt,
    # this might be useful. For now, we assume the `generator.py` handles extracting only the new text.
    
    # A more generic cleaner might look for phrases like "New Journal Entry:" if they accidentally get included.
    # For now, just strip whitespace.
    text = text.strip()
    return text

def count_words(text: str) -> int:
    """Counts the number of words in a text using nltk.word_tokenize."""
    if not text:
        return 0
    return len(text.split())

def check_word_count_adherence(text_word_count: int, target_word_count: int, tolerance_percentage: float = 0.50) -> tuple[bool, float]:
    """
    Checks if the actual word count is within a tolerance percentage of the target word count.

    Args:
        text_word_count (int): The actual word count of the generated text.
        target_word_count (int): The desired word count.
        tolerance_percentage (float): The allowable deviation (e.g., 0.50 for 50%).

    Returns:
        tuple[bool, float]: (is_adherent, deviation_percentage)
                          is_adherent is True if within tolerance.
                          deviation_percentage is the actual signed deviation from target.
    """
    if target_word_count == 0: # Avoid division by zero if target is 0 for some reason
        return text_word_count == 0, 0.0

    lower_bound = target_word_count * (1 - tolerance_percentage)
    upper_bound = target_word_count * (1 + tolerance_percentage)
    
    is_adherent = lower_bound <= text_word_count <= upper_bound
    deviation = (text_word_count - target_word_count) / target_word_count
    return is_adherent, deviation

def smart_truncate_text(text: str, target_word_count: int, max_overshoot_words: int = 30) -> str:
    """
    Truncates text to be close to the target_word_count.
    If the text is already shorter or slightly longer (within max_overshoot_words), it's returned as is.
    If longer, it truncates to the target_word_count, trying to preserve sentence endings if the cut is close.
    This is a simplified version; true sentence-aware truncation can be more complex.

    Args:
        text (str): The text to truncate.
        target_word_count (int): The desired word count after truncation.
        max_overshoot_words (int): How many words over the target_word_count is acceptable without truncation.

    Returns:
        str: The potentially truncated text.
    """
    words = word_tokenize(text)
    current_word_count = len(words)

    if current_word_count <= target_word_count + max_overshoot_words:
        return text # Already within acceptable length or shorter

    # Simple truncation to target_word_count for now if significantly over.
    # More advanced: could look for last sentence end before target_word_count + small_buffer
    # For now, we'll just cut to the target word count if it's too long.
    truncated_words = words[:target_word_count]
    # Re-join carefully, especially with punctuation that word_tokenize might separate.
    # A simple join might not be perfect for all tokenized outputs.
    # Using nltk.tokenize.treebank.TreebankWordDetokenizer would be more robust but adds dependency/complexity.
    # For now, a simple join is often good enough.
    # However, NLTK's `word_tokenize` often splits contractions and punctuation. `detokenize` is better.
    # Since we don't have it here, a simple join. User should be aware of potential minor spacing issues.
    return " ".join(truncated_words).strip() # Basic rejoining

if __name__ == '__main__':
    print("--- Utils Test ---")
    current_date = get_current_datetime_str_for_file_id()
    print(f"Current date string: {current_date}")

    # file_id_1 = generate_file_id(1, 10)
    # file_id_10 = generate_file_id(10, 10)
    # file_id_5_of_100 = generate_file_id(5, 100)
    # print(f"File ID (1 of 10): {file_id_1}")
    # print(f"File ID (10 of 10): {file_id_10}")
    # print(f"File ID (5 of 100): {file_id_5_of_100}")

    filename = construct_filename(current_date)
    print(f"Constructed filename: {filename}")

    raw_text = "  New Journal Entry: This is a test. It has some words.  \nThis is another sentence. "
    cleaned = clean_generated_text(raw_text)
    print(f"Original text: '{raw_text}'")
    print(f"Cleaned text: '{cleaned}'")

    long_text = "This is a very long journal entry that needs to be truncated. It has many words, far more than we actually want for this particular example. We will see how the truncation function handles this situation. Hopefully, it does a reasonable job. We are aiming for about 20 words. This sentence makes it much longer."
    word_tokens = word_tokenize(long_text)
    print(f"Original word count for long_text: {len(word_tokens)}")
    
    # Test check_word_count_adherence
    print("\n--- Testing Word Count Adherence ---")
    target_wc = 100
    actual_wc_good = 90 # Within 20% (old default), within 50% (new default)
    actual_wc_bad_low = 70 # Within 20% (old default), within 50% (new default)
    actual_wc_bad_high = 130 # Outside 20% (old default), within 50% (new default)
    adherent, dev = check_word_count_adherence(actual_wc_good, target_wc)
    print(f"Target: {target_wc}, Actual: {actual_wc_good} -> Adherent: {adherent}, Deviation: {dev:.2%}")
    adherent, dev = check_word_count_adherence(actual_wc_bad_low, target_wc)
    print(f"Target: {target_wc}, Actual: {actual_wc_bad_low} -> Adherent: {adherent}, Deviation: {dev:.2%}")
    adherent, dev = check_word_count_adherence(actual_wc_bad_high, target_wc)
    print(f"Target: {target_wc}, Actual: {actual_wc_bad_high} -> Adherent: {adherent}, Deviation: {dev:.2%}")

    # Test smart_truncate_text (formerly truncate_to_word_count)
    print("\n--- Testing Smart Truncation ---")
    long_text_for_trunc = "This is sentence one. This is sentence two, which is a bit longer. Sentence three is the final one here for this test example."
    words_in_long_text = count_words(long_text_for_trunc)
    print(f"Original for truncation: '{long_text_for_trunc}' (Words: {words_in_long_text})")

    truncated_1 = smart_truncate_text(long_text_for_trunc, 15, max_overshoot_words=5)
    print(f"Truncated (target 15, overshoot 5): '{truncated_1}' (Words: {count_words(truncated_1)})")
    
    truncated_2 = smart_truncate_text(long_text_for_trunc, 10, max_overshoot_words=2)
    print(f"Truncated (target 10, overshoot 2): '{truncated_2}' (Words: {count_words(truncated_2)})")

    # Text that is already short enough
    short_text = "This is short."
    truncated_short = smart_truncate_text(short_text, 10, max_overshoot_words=2)
    print(f"Truncated short (target 10): '{truncated_short}' (Words: {count_words(truncated_short)})") 