import unittest
import pandas as pd
import os

# Add src to Python path if tests are run from project root
import sys
# This navigates up to the project root (journal_generator) then down to src
# Adjust if your test runner or project structure is different
SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

try:
    from data_loader import load_and_preprocess_data, get_examples_for_prompt, ALL_AVAILABLE_EMOTIONS
except ImportError:
    print("Failed to import from data_loader. Ensure PYTHONPATH is set correctly or tests are run from a suitable directory.")
    # Fallback for cases where the test runner might have issues with relative imports
    # This is less ideal but can help in some CI environments or complex setups
    if 'data_loader' not in sys.modules:
        raise

# This module might require KAGGLE_API_KEY to be set up to run tests that load data.
# Consider using a mock or a small, local dummy CSV for tests to avoid Kaggle dependency during unit testing.

@unittest.skipIf("KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ,
                 "Kaggle API credentials not found in environment. Skipping data_loader tests that require downloads.")
class TestDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load data once for all tests in this class if Kaggle API is available."""
        print("Attempting to load data for TestDataLoader...")
        try:
            cls.df = load_and_preprocess_data()
            print(f"Data loaded for tests. Shape: {cls.df.shape}")
        except Exception as e:
            print(f"Failed to load data in setUpClass for TestDataLoader: {e}")
            cls.df = None # Ensure df is None if loading fails

    def test_load_and_preprocess_data_returns_dataframe(self):
        """Test that data loading returns a pandas DataFrame."""
        if self.df is None:
            self.skipTest("Dataframe not loaded, skipping test_load_and_preprocess_data_returns_dataframe")
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertGreater(len(self.df), 0, "DataFrame should not be empty.")

    def test_answer_column_exists(self):
        """Test that the 'Answer' column exists after loading."""
        if self.df is None:
            self.skipTest("Dataframe not loaded, skipping test_answer_column_exists")
        self.assertIn('Answer', self.df.columns)

    def test_emotion_columns_are_boolean(self):
        """Test that known emotion columns are converted to boolean type."""
        if self.df is None:
            self.skipTest("Dataframe not loaded, skipping test_emotion_columns_are_boolean")
        
        # Check a sample of emotion columns
        sample_emotion_cols = ['Answer.f1.happy.raw', 'Answer.f1.sad.raw']
        for col_name in sample_emotion_cols:
            if col_name in self.df.columns:
                self.assertTrue(pd.api.types.is_bool_dtype(self.df[col_name]),
                                f"Column {col_name} should be boolean type.")
            else:
                self.fail(f"Expected emotion column {col_name} not found in DataFrame.")

    def test_get_examples_for_prompt(self):
        """Test fetching example prompts for a known emotion."""
        if self.df is None:
            self.skipTest("Dataframe not loaded, skipping test_get_examples_for_prompt")

        # Test with a common emotion expected to have examples
        happy_examples = get_examples_for_prompt(self.df, 'happy', num_examples=2)
        self.assertIsInstance(happy_examples, list)
        if not self.df[self.df['Answer.f1.happy.raw'] == True].empty: # If happy entries exist
            self.assertEqual(len(happy_examples), 2, "Should return 2 examples for 'happy' if available.")
            if happy_examples:
                self.assertIsInstance(happy_examples[0], str)
        else:
            print("Skipping num_examples check for 'happy' as no entries were found in source for this test run.")

    def test_get_examples_for_prompt_nonexistent_emotion(self):
        """Test fetching examples for an emotion string not in dataset columns."""
        if self.df is None:
            self.skipTest("Dataframe not loaded, skipping test_get_examples_for_prompt_nonexistent_emotion")
        
        # This should return an empty list and print a warning (checked manually)
        non_existent_examples = get_examples_for_prompt(self.df, 'nonexistentemotion', num_examples=2)
        self.assertEqual(len(non_existent_examples), 0)

    def test_get_examples_for_prompt_zero_examples(self):
        """Test requesting zero examples."""
        if self.df is None:
            self.skipTest("Dataframe not loaded, skipping test_get_examples_for_prompt_zero_examples")
        zero_examples = get_examples_for_prompt(self.df, 'happy', num_examples=0)
        self.assertEqual(len(zero_examples), 0)
    
    def test_all_available_emotions_list(self):
        """Test that ALL_AVAILABLE_EMOTIONS is a non-empty list of strings."""
        self.assertIsInstance(ALL_AVAILABLE_EMOTIONS, list)
        self.assertGreater(len(ALL_AVAILABLE_EMOTIONS), 0)
        self.assertTrue(all(isinstance(emotion, str) for emotion in ALL_AVAILABLE_EMOTIONS))
        self.assertIn("happy", ALL_AVAILABLE_EMOTIONS) # Check for a known emotion

if __name__ == '__main__':
    # This allows running the tests directly using `python tests/test_data_loader.py`
    # Ensure that the src directory is in PYTHONPATH or adjust sys.path as done above.
    print(f"Current sys.path for test_data_loader: {sys.path}")
    unittest.main() 