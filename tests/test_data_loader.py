import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import os
import sys

# Add src to Python path
SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# It's crucial that this import happens *after* sys.path is potentially modified,
# and that data_loader.py itself doesn't try to load data at import time in a way
# that would break tests if the real data file isn't there.
from data_loader import load_and_preprocess_data, get_examples_for_prompt, ALL_AVAILABLE_EMOTIONS, EMOTION_COLUMNS, LOCAL_DATA_FILE

class TestDataLoader(unittest.TestCase):

    def _create_sample_df(self, data_dict=None):
        """Helper to create a sample DataFrame for testing."""
        if data_dict is None:
            data_dict = {
                'Answer': ["Entry 1", "Entry 2", "Entry 3", "Entry 4", "Entry 5"],
                'Answer.f1.happy.raw': ['TRUE', 'FALSE', 'TRUE', 'FALSE', 'TRUE'],
                'Answer.f1.sad.raw': ['FALSE', 'TRUE', 'FALSE', 'TRUE', 'FALSE'],
                'Answer.f1.anxious.raw': ['TRUE', 'TRUE', 'FALSE', 'FALSE', 'NAN'], # Test NAN
                'Answer.f1.proud.raw': [1, 0, 1, 0, 0] 
            }
        return pd.DataFrame(data_dict)

    @patch('data_loader.pd.read_csv')
    @patch('data_loader.os.path.exists')
    def test_load_and_preprocess_data_success(self, mock_exists, mock_read_csv):
        """Test successful loading and preprocessing of data."""
        mock_exists.return_value = True
        sample_df = self._create_sample_df()
        mock_read_csv.return_value = sample_df.copy() # Use a copy

        df = load_and_preprocess_data()

        mock_exists.assert_called_once_with(LOCAL_DATA_FILE)
        mock_read_csv.assert_called_once_with(LOCAL_DATA_FILE)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('Answer', df.columns)
        
        # Test boolean conversion
        self.assertTrue(pd.api.types.is_bool_dtype(df['Answer.f1.happy.raw']))
        self.assertEqual(df['Answer.f1.happy.raw'].tolist(), [True, False, True, False, True])
        
        self.assertTrue(pd.api.types.is_bool_dtype(df['Answer.f1.sad.raw']))
        self.assertEqual(df['Answer.f1.sad.raw'].tolist(), [False, True, False, True, False])

        self.assertTrue(pd.api.types.is_bool_dtype(df['Answer.f1.anxious.raw']))
        self.assertEqual(df['Answer.f1.anxious.raw'].tolist(), [True, True, False, False, False]) # NAN becomes False

        # Test numeric to boolean conversion for a column defined in EMOTION_COLUMNS
        self.assertTrue(pd.api.types.is_bool_dtype(df['Answer.f1.proud.raw']))
        self.assertEqual(df['Answer.f1.proud.raw'].tolist(), [True, False, True, False, False])
        
        # Test Answer column stripping (if applicable, though not explicitly in current data_loader)
        # For now, just check it's string
        self.assertTrue(all(isinstance(x, str) for x in df['Answer']))


    @patch('data_loader.os.path.exists')
    def test_load_and_preprocess_data_file_not_found(self, mock_exists):
        """Test FileNotFoundError when data file doesn't exist."""
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            load_and_preprocess_data()
        mock_exists.assert_called_once_with(LOCAL_DATA_FILE)

    @patch('data_loader.pd.read_csv')
    @patch('data_loader.os.path.exists')
    def test_load_and_preprocess_data_missing_answer_column(self, mock_exists, mock_read_csv):
        """Test KeyError if 'Answer' column is missing."""
        mock_exists.return_value = True
        # Create a DataFrame without the 'Answer' column
        bad_data = {
            'Answer.f1.happy.raw': ['TRUE', 'FALSE'],
            # Missing 'Answer' column
        }
        mock_read_csv.return_value = pd.DataFrame(bad_data)
        with self.assertRaises(KeyError):
            load_and_preprocess_data()

    @patch('data_loader.pd.read_csv')
    @patch('data_loader.os.path.exists')
    def test_load_and_preprocess_data_handles_missing_emotion_cols(self, mock_exists, mock_read_csv):
        """Test that missing emotion columns are handled gracefully (warning printed)."""
        mock_exists.return_value = True
        # Create a DataFrame with only 'Answer' and one emotion column
        partial_data = {
            'Answer': ["Entry 1", "Entry 2"],
            'Answer.f1.happy.raw': ['TRUE', 'FALSE'],
            # Other EMOTION_COLUMNS are missing
        }
        mock_df = pd.DataFrame(partial_data)
        mock_read_csv.return_value = mock_df.copy()

        # Patch print to capture warnings
        with patch('builtins.print') as mock_print:
            df = load_and_preprocess_data()
        
        self.assertIn('Answer.f1.happy.raw', df.columns)
        self.assertTrue(pd.api.types.is_bool_dtype(df['Answer.f1.happy.raw']))
        
        # Check that warnings were printed for missing columns
        # This depends on the exact warning message in data_loader.py
        missing_col_warnings = 0
        for expected_col in EMOTION_COLUMNS:
            if expected_col not in partial_data: # If it was truly missing from input
                 # Check if a warning about this specific column was printed
                self.assertTrue(any(f"Warning: Emotion column {expected_col} not found" in call_args.args[0] 
                                   for call_args in mock_print.call_args_list),
                                   f"Expected warning for missing column {expected_col} not found.")
                missing_col_warnings +=1
        self.assertGreater(missing_col_warnings, 0, "Should have printed warnings for missing emotion columns")


    def test_get_examples_for_prompt_success(self):
        """Test successfully fetching examples."""
        sample_df = self._create_sample_df({
            'Answer': ["Happy Day", "Joyful Times", "Sunny Entry", "Sad Story", "Blue Mood"],
            'Answer.f1.happy.raw': [True, True, True, False, False],
            'Answer.f1.sad.raw': [False, False, False, True, True]
        })
        examples = get_examples_for_prompt(sample_df, 'happy', num_examples=2)
        self.assertEqual(len(examples), 2)
        for ex in examples:
            self.assertIn(ex, ["Happy Day", "Joyful Times", "Sunny Entry"])

        # Test case insensitivity for emotion
        examples_case = get_examples_for_prompt(sample_df, 'HAPPY', num_examples=1)
        self.assertEqual(len(examples_case), 1)
        self.assertIn(examples_case[0], ["Happy Day", "Joyful Times", "Sunny Entry"])


    def test_get_examples_for_prompt_fewer_than_requested(self):
        """Test fetching when fewer examples exist than requested."""
        sample_df = self._create_sample_df({
            'Answer': ["One Happy Entry", "Other stuff"],
            'Answer.f1.happy.raw': [True, False],
            'Answer.f1.sad.raw': [False, True]
        })
        with patch('builtins.print') as mock_print: # To check for warning
            examples = get_examples_for_prompt(sample_df, 'happy', num_examples=3)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0], "One Happy Entry")
        self.assertTrue(any("Warning: Found only 1 entries for emotion 'happy'" in call_args.args[0] 
                            for call_args in mock_print.call_args_list))

    def test_get_examples_for_prompt_no_examples_found(self):
        """Test fetching when no examples exist for the emotion."""
        sample_df = self._create_sample_df({
            'Answer': ["All Sad Here", "Very Blue"],
            'Answer.f1.happy.raw': [False, False], # No happy entries
            'Answer.f1.sad.raw': [True, True]
        })
        with patch('builtins.print') as mock_print: # To check for message
            examples = get_examples_for_prompt(sample_df, 'happy', num_examples=2)
        self.assertEqual(len(examples), 0)
        self.assertTrue(any("No entries found for emotion: happy" in call_args.args[0]
                            for call_args in mock_print.call_args_list))

    def test_get_examples_for_prompt_nonexistent_emotion_column(self):
        """Test fetching for an emotion whose column doesn't exist in the DataFrame."""
        sample_df = self._create_sample_df() # Standard columns
        with patch('builtins.print') as mock_print: # To check for warning
            examples = get_examples_for_prompt(sample_df, 'nonexistentemotion', num_examples=2)
        self.assertEqual(len(examples), 0)
        self.assertTrue(any("Warning: Emotion column for 'nonexistentemotion' (Answer.f1.nonexistentemotion.raw) not found" 
                            in call_args.args[0] for call_args in mock_print.call_args_list))

    def test_get_examples_for_prompt_zero_examples_requested(self):
        """Test requesting zero examples returns an empty list."""
        sample_df = self._create_sample_df()
        examples = get_examples_for_prompt(sample_df, 'happy', num_examples=0)
        self.assertEqual(len(examples), 0)

    def test_get_examples_for_prompt_emotion_column_not_boolean(self):
        """Test behavior when the target emotion column is not boolean and cannot be converted."""
        sample_df = pd.DataFrame({
            'Answer': ["Entry A", "Entry B"],
            'Answer.f1.weird.raw': ["Yes", "No"] # Not TRUE/FALSE, not 0/1
        })
        with patch('builtins.print') as mock_print:
            examples = get_examples_for_prompt(sample_df, 'weird', num_examples=1)
        self.assertEqual(len(examples), 0)
        self.assertTrue(any("Warning: Emotion column Answer.f1.weird.raw is not boolean." in call_args.args[0]
                            for call_args in mock_print.call_args_list))

    def test_all_available_emotions_list(self):
        """Test that ALL_AVAILABLE_EMOTIONS is a non-empty list of strings."""
        self.assertIsInstance(ALL_AVAILABLE_EMOTIONS, list)
        self.assertGreater(len(ALL_AVAILABLE_EMOTIONS), 0)
        self.assertTrue(all(isinstance(emotion, str) for emotion in ALL_AVAILABLE_EMOTIONS))
        # Check for a few known emotions based on EMOTION_COLUMNS
        self.assertIn("happy", ALL_AVAILABLE_EMOTIONS)
        self.assertIn("sad", ALL_AVAILABLE_EMOTIONS)
        self.assertIn("afraid", ALL_AVAILABLE_EMOTIONS)
        # Check a more complex one to ensure splitting worked
        self.assertTrue(any(col.split('.')[2] == "surprised" for col in EMOTION_COLUMNS))
        self.assertIn("surprised", ALL_AVAILABLE_EMOTIONS)


if __name__ == '__main__':
    unittest.main() 