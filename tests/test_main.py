import unittest
from unittest.mock import patch, call
import sys
import os
from argparse import Namespace # For creating mock args

# Add the src directory to the Python path to allow imports from main
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "src")
sys.path.insert(0, src_dir)

# Import main function and constants
try:
    from main import main, DEFAULT_OUTPUT_DIR, ALL_AVAILABLE_EMOTIONS
except ImportError:
    # This might happen if script is not run from project root or venv not active
    # Fallback for simpler test execution if needed, assuming src is in PYTHONPATH
    print("Attempting fallback import for main due to potential PYTHONPATH issue in test runner.")
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add project root
    from src.main import main, DEFAULT_OUTPUT_DIR, ALL_AVAILABLE_EMOTIONS


class TestMainScriptArguments(unittest.TestCase):

    def common_mocks(self):
        """Return a dictionary of common mocks needed for most tests."""
        # Mock modules that main.py interacts with
        mock_load_dotenv = patch('main.load_dotenv') # if main directly calls it
        mock_load_and_preprocess_data = patch('main.load_and_preprocess_data')
        mock_get_examples_for_prompt = patch('main.get_examples_for_prompt')
        mock_journal_generator_class = patch('main.JournalGenerator')
        mock_journal_exporter_class = patch('main.JournalExporter')
        
        # Start patches and return their mock objects
        mocks = {
            'load_dotenv': mock_load_dotenv.start(),
            'load_data': mock_load_and_preprocess_data.start(),
            'get_examples': mock_get_examples_for_prompt.start(),
            'GeneratorClass': mock_journal_generator_class.start(),
            'ExporterClass': mock_journal_exporter_class.start(),
        }
        
        # Configure default return values for mocked instances/methods
        # Mock the JournalGenerator instance and its generate_entry method
        self.mock_generator_instance = mocks['GeneratorClass'].return_value
        self.mock_generator_instance.generate_entry.return_value = "Mocked journal entry"

        # Mock the JournalExporter instance and its save_entry method
        self.mock_exporter_instance = mocks['ExporterClass'].return_value
        self.mock_exporter_instance.save_entry.return_value = "mocked_entry.txt"
        
        # Mock data loading to return a dummy DataFrame or None
        mocks['load_data'].return_value = "dummy_dataframe" # Simulate successful load
        mocks['get_examples'].return_value = ["example 1", "example 2"]

        self.addCleanup(patch.stopall) # Ensure all patches are stopped after each test
        return mocks

    def test_default_arguments(self):
        """Test main() with only the required --tone argument, checking defaults."""
        mocks = self.common_mocks()
        
        test_tone = ALL_AVAILABLE_EMOTIONS[0] if ALL_AVAILABLE_EMOTIONS else "happy"
        cli_args = ['main.py', '--tone', test_tone]

        with patch.object(sys, 'argv', cli_args):
            main()

        mocks['GeneratorClass'].assert_called_once_with()
        mocks['ExporterClass'].assert_called_once_with(output_dir=DEFAULT_OUTPUT_DIR)
        self.mock_generator_instance.generate_entry.assert_called_once()
        call_args = self.mock_generator_instance.generate_entry.call_args
        self.assertEqual(call_args.kwargs['target_emotion'], test_tone)
        self.assertEqual(call_args.kwargs['avg_word_count'], 100)
        self.assertEqual(len(call_args.kwargs['example_entries']), 2)
        self.assertEqual(call_args.kwargs['max_new_tokens'], 0)
        mocks['load_data'].assert_called_once()
        mocks['get_examples'].assert_called_once()
        self.mock_exporter_instance.save_entry.assert_called_once_with(entry_text="Mocked journal entry")

    def test_custom_arguments(self):
        """Test main() with various custom arguments."""
        mocks = self.common_mocks()
        
        test_tone = ALL_AVAILABLE_EMOTIONS[1] if len(ALL_AVAILABLE_EMOTIONS) > 1 else "sad"
        cli_args = [
            'main.py', 
            '--num_days', '2',
            '--entries_per_day', '1',
            '--avg_word_count', '150',
            '--tone', test_tone,
            '--output_dir', 'custom_output',
            '--num_examples_prompt', '2',
            '--max_generation_tokens', '200',
            '--start_date', '20230101'
        ]
        with patch.object(sys, 'argv', cli_args):
            main()

        mocks['ExporterClass'].assert_called_once_with(output_dir='custom_output')
        self.assertEqual(self.mock_generator_instance.generate_entry.call_count, 2 * 1)
        first_call_args = self.mock_generator_instance.generate_entry.call_args_list[0]
        self.assertEqual(first_call_args.kwargs['target_emotion'], test_tone)
        self.assertEqual(first_call_args.kwargs['avg_word_count'], 150)
        self.assertEqual(len(first_call_args.kwargs['example_entries']), 2)
        self.assertEqual(first_call_args.kwargs['max_new_tokens'], 200)
        mocks['load_data'].assert_called_once()
        self.assertEqual(mocks['get_examples'].call_count, 2 * 1)
        self.assertEqual(mocks['get_examples'].call_args.args[2], 2)

    def test_disable_examples(self):
        """Test main() with --num_examples_prompt 0."""
        mocks = self.common_mocks()
        
        test_tone = ALL_AVAILABLE_EMOTIONS[0] if ALL_AVAILABLE_EMOTIONS else "neutral"
        cli_args = ['main.py', '--tone', test_tone, '--num_examples_prompt', '0']
        with patch.object(sys, 'argv', cli_args):
            main()

        mocks['load_data'].assert_not_called()
        mocks['get_examples'].assert_not_called()
        call_args = self.mock_generator_instance.generate_entry.call_args
        self.assertEqual(call_args.kwargs['example_entries'], [])

    @patch('builtins.print')
    def test_missing_tone_argument(self, mock_print):
        """Test that argparse exits if --tone is missing."""
        mocks = self.common_mocks()
        
        cli_args = ['main.py', '--num_days', '1']
        with patch.object(sys, 'argv', cli_args):
            with self.assertRaises(SystemExit):
                main()

    def test_invalid_tone_choice(self):
        """Test that argparse exits if --tone is invalid."""
        mocks = self.common_mocks()
        
        cli_args = ['main.py', '--tone', 'non_existent_emotion']
        with patch.object(sys, 'argv', cli_args):
            with self.assertRaises(SystemExit):
                main()

    def test_data_loading_failure(self):
        """Test how main handles failure in load_and_preprocess_data."""
        mocks = self.common_mocks()
        
        mocks['load_data'].side_effect = Exception("Failed to load CSV")
        
        test_tone = ALL_AVAILABLE_EMOTIONS[0] if ALL_AVAILABLE_EMOTIONS else "curious"
        cli_args = ['main.py', '--tone', test_tone, '--num_examples_prompt', '1']
        
        with patch.object(sys, 'argv', cli_args):
            main()
        
        mocks['load_data'].assert_called_once()
        mocks['get_examples'].assert_not_called()
        
        call_args = self.mock_generator_instance.generate_entry.call_args
        self.assertEqual(call_args.kwargs['example_entries'], [])

if __name__ == '__main__':
    unittest.main() 