import unittest
import os
import shutil  # For easily removing a directory and its contents
import sys
from unittest.mock import patch

SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

try:
    from exporter import JournalExporter
    from utils import get_current_datetime_str, generate_file_id, construct_filename
except ImportError:
    print("Failed to import from exporter or utils. Ensure PYTHONPATH is set correctly.")
    if 'exporter' not in sys.modules or 'utils' not in sys.modules:
        raise

class TestJournalExporter(unittest.TestCase):
    TEST_OUTPUT_DIR = "test_generated_entries_temp"

    def setUp(self):
        """Create a temporary directory for test output files."""
        # Ensure a clean state by removing the directory if it exists from a previous failed run
        if os.path.exists(self.TEST_OUTPUT_DIR):
            shutil.rmtree(self.TEST_OUTPUT_DIR)
        os.makedirs(self.TEST_OUTPUT_DIR, exist_ok=True)
        self.exporter = JournalExporter(output_dir=self.TEST_OUTPUT_DIR)

    def tearDown(self):
        """Remove the temporary directory and its contents after tests."""
        if os.path.exists(self.TEST_OUTPUT_DIR):
            shutil.rmtree(self.TEST_OUTPUT_DIR)

    def test_exporter_initialization_creates_directory(self):
        """Test that the exporter creates the output directory if it doesn't exist."""
        # setUp already creates it, so we test by re-creating exporter for a new sub-directory
        new_dir = os.path.join(self.TEST_OUTPUT_DIR, "new_sub_dir")
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        self.assertFalse(os.path.exists(new_dir))
        JournalExporter(output_dir=new_dir) # Initialize to trigger directory creation
        self.assertTrue(os.path.exists(new_dir), "Exporter should create the output directory.")

    def test_save_entry_creates_file(self):
        """Test that save_entry creates a file with the correct name and content."""
        entry_text = "This is a test journal entry for the exporter."
        date_str = "20231101"
        entry_index = 1
        total_entries = 5

        expected_file_id = generate_file_id(entry_index, total_entries)
        expected_filename = construct_filename(date_str, expected_file_id)
        expected_filepath = os.path.join(self.TEST_OUTPUT_DIR, expected_filename)

        saved_filepath = self.exporter.save_entry(entry_text, date_str, entry_index, total_entries)
        
        self.assertEqual(saved_filepath, expected_filepath)
        self.assertTrue(os.path.exists(expected_filepath), "File should be created.")

        with open(expected_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, entry_text, "File content does not match the entry text.")

    def test_save_entry_handles_empty_text(self):
        """Test that save_entry does not create a file for empty text and returns None."""
        date_str = "20231101"
        entry_index = 2
        total_entries = 5
        
        saved_filepath = self.exporter.save_entry("", date_str, entry_index, total_entries)
        self.assertIsNone(saved_filepath, "Should return None for empty entry text.")
        
        # Check that no unexpected file was created
        # (Relies on knowing the naming convention, but that's okay for this test)
        potential_file_id = generate_file_id(entry_index, total_entries)
        potential_filename = construct_filename(date_str, potential_file_id)
        potential_filepath = os.path.join(self.TEST_OUTPUT_DIR, potential_filename)
        self.assertFalse(os.path.exists(potential_filepath), "No file should be created for an empty entry.")

    def test_save_entry_uses_correct_padding_for_file_id(self):
        """Test filename padding based on total_entries_for_day."""
        entry_text = "Testing padding."
        date_str = "20231102"
        
        # Test with 1 entry total (e.g., journal_DATE_1.txt)
        saved_filepath_1 = self.exporter.save_entry(entry_text, date_str, 1, 1)
        self.assertIsNotNone(saved_filepath_1)
        self.assertTrue(os.path.basename(saved_filepath_1).endswith("_1.txt"))

        # Test with 10 entries total (e.g., journal_DATE_01.txt, journal_DATE_10.txt)
        saved_filepath_01 = self.exporter.save_entry(entry_text, date_str, 1, 10)
        self.assertIsNotNone(saved_filepath_01)
        self.assertTrue(os.path.basename(saved_filepath_01).endswith("_01.txt"))
        
        saved_filepath_10 = self.exporter.save_entry(entry_text, date_str, 10, 10)
        self.assertIsNotNone(saved_filepath_10)
        self.assertTrue(os.path.basename(saved_filepath_10).endswith("_10.txt"))

        # Test with 100 entries total (e.g., journal_DATE_001.txt)
        saved_filepath_001 = self.exporter.save_entry(entry_text, date_str, 1, 100)
        self.assertIsNotNone(saved_filepath_001)
        self.assertTrue(os.path.basename(saved_filepath_001).endswith("_001.txt"))

    def test_save_entry_io_error(self):
        """Test how save_entry handles an IOError (e.g., invalid path)."""
        # We can mock open to raise an IOError
        with patch('builtins.open', side_effect=IOError("Disk full simulation")):
            saved_filepath = self.exporter.save_entry("text", "20230101", 1, 1)
            self.assertIsNone(saved_filepath, "Should return None on IOError.")


if __name__ == '__main__':
    unittest.main() 