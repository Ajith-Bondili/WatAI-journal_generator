import unittest
import os
import shutil
import sys
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "src")
sys.path.insert(0, src_dir)

from exporter import JournalExporter
# utils.construct_filename will still be used by the exporter, but its input is from the mocked timestamp func
# utils.generate_file_id and utils.get_current_datetime_str are no longer used by exporter for ID generation
import utils # Keep for patching its method

class TestJournalExporter(unittest.TestCase):
    TEST_OUTPUT_DIR = "test_generated_entries_temp_timestamp"

    def setUp(self):
        """Create a temporary directory for test output files."""
        if os.path.exists(self.TEST_OUTPUT_DIR):
            shutil.rmtree(self.TEST_OUTPUT_DIR)
        # Exporter will create the directory
        # os.makedirs(self.TEST_OUTPUT_DIR, exist_ok=True)
        self.exporter = JournalExporter(output_dir=self.TEST_OUTPUT_DIR)

    def tearDown(self):
        """Remove the temporary directory and its contents after tests."""
        if os.path.exists(self.TEST_OUTPUT_DIR):
            shutil.rmtree(self.TEST_OUTPUT_DIR)

    def test_exporter_initialization_creates_directory(self):
        """Test that the exporter creates the output directory if it doesn't exist."""
        new_dir = os.path.join(self.TEST_OUTPUT_DIR, "new_sub_dir_ts")
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        self.assertFalse(os.path.exists(new_dir))
        JournalExporter(output_dir=new_dir) # Initialize to trigger directory creation
        self.assertTrue(os.path.exists(new_dir), "Exporter should create the output directory.")

    @patch('utils.get_current_datetime_str_for_file_id')
    def test_save_entry_creates_file_with_timestamp_id(self, mock_get_timestamp_id):
        """Test that save_entry creates a file with a mocked timestamp-based ID and correct content."""
        entry_text = "This is a test journal entry with a timestamp ID."
        mocked_timestamp_id = "20231101_100000123456"
        mock_get_timestamp_id.return_value = mocked_timestamp_id
        
        expected_filename = utils.construct_filename(mocked_timestamp_id) # Uses the new construct_filename signature
        expected_filepath = os.path.join(self.TEST_OUTPUT_DIR, expected_filename)

        saved_filepath = self.exporter.save_entry(entry_text)
        
        mock_get_timestamp_id.assert_called_once()
        self.assertEqual(saved_filepath, expected_filepath)
        self.assertTrue(os.path.exists(expected_filepath), "File should be created.")

        with open(expected_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, entry_text, "File content does not match the entry text.")

    def test_save_entry_handles_empty_text_timestamp(self):
        """Test that save_entry does not create a file for empty text and returns None (timestamp version)."""
        with patch('utils.get_current_datetime_str_for_file_id') as mock_get_timestamp_id:
            saved_filepath = self.exporter.save_entry("")
            self.assertIsNone(saved_filepath, "Should return None for empty entry text.")
            mock_get_timestamp_id.assert_not_called() # Should not try to get a timestamp if text is empty

    @patch('utils.get_current_datetime_str_for_file_id')
    def test_multiple_saves_create_unique_files_with_mocked_timestamps(self, mock_get_timestamp_id):
        """Test that multiple calls to save_entry with different mocked timestamps create unique files."""
        entry_text_1 = "First entry."
        entry_text_2 = "Second entry."
        timestamp_1 = "20231101_110000000000"
        timestamp_2 = "20231101_110000111111"

        # Configure mock to return different values on subsequent calls
        mock_get_timestamp_id.side_effect = [timestamp_1, timestamp_2]

        # First save
        saved_filepath_1 = self.exporter.save_entry(entry_text_1)
        expected_filename_1 = utils.construct_filename(timestamp_1)
        self.assertIsNotNone(saved_filepath_1)
        self.assertEqual(os.path.basename(saved_filepath_1), expected_filename_1)
        self.assertTrue(os.path.exists(saved_filepath_1))

        # Second save
        saved_filepath_2 = self.exporter.save_entry(entry_text_2)
        expected_filename_2 = utils.construct_filename(timestamp_2)
        self.assertIsNotNone(saved_filepath_2)
        self.assertEqual(os.path.basename(saved_filepath_2), expected_filename_2)
        self.assertTrue(os.path.exists(saved_filepath_2))

        self.assertNotEqual(saved_filepath_1, saved_filepath_2, "Filenames should be unique.")
        self.assertEqual(mock_get_timestamp_id.call_count, 2)

    def test_save_entry_io_error_timestamp(self):
        """Test how save_entry handles an IOError (timestamp version)."""
        with patch('utils.get_current_datetime_str_for_file_id', return_value="20231101_120000000000") as mock_ts:
            with patch('builtins.open', side_effect=IOError("Disk full simulation")) as mock_open:
                saved_filepath = self.exporter.save_entry("text")
                self.assertIsNone(saved_filepath, "Should return None on IOError.")
                mock_ts.assert_called_once() # Timestamp should still be generated
                mock_open.assert_called_once() # Open should have been attempted

if __name__ == '__main__':
    unittest.main() 