import os
# from datetime import datetime # No longer needed directly here
# import json # No longer needed for counter

# Try to import from local src package first, then parent directory if running script directly
if __package__:
    from . import utils
else:
    import utils # For standalone testing if necessary

class JournalExporter:
    # _COUNTER_FILE_PATH, _load_counter, _save_counter are removed

    def __init__(self, output_dir: str = "generated_entries"):
        """
        Initializes the journal exporter.

        Args:
            output_dir (str): The directory where generated journal entries will be saved.
        """
        self.output_dir = output_dir
        # Ensure the output directory exists
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Output directory '{self.output_dir}' ensured.")
        except OSError as e:
            print(f"Error creating output directory '{self.output_dir}': {e}")
            raise
        # Removed self.global_counter initialization
        print(f"Exporter initialized. Output will be saved to: {os.path.abspath(self.output_dir)}.")

    # _load_counter and _save_counter methods are removed

    def save_entry(
        self,
        entry_text: str
        # date_str, entry_index_in_day, total_entries_for_day are removed
    ) -> str | None:
        """
        Saves a single journal entry to a .txt file using a unique timestamp ID.

        Args:
            entry_text (str): The content of the journal entry.

        Returns:
            str | None: The full path to the saved file if successful, None otherwise.
        """
        if not entry_text:
            print("Warning: Attempted to save an empty entry. Skipping.")
            return None

        # Removed self.global_counter increment
        unique_id_str = utils.get_current_datetime_str_for_file_id()
        filename = utils.construct_filename(unique_id_str) # Prefix defaults to "journal"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(entry_text)
            
            # Removed self._save_counter()
            print(f"Journal entry saved to: {filepath}")
            return filepath
        except IOError as e:
            print(f"Error saving journal entry to {filename}: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during save_entry: {e}")
            return None

if __name__ == '__main__':
    print("Testing JournalExporter with Timestamp-based IDs...")
    test_output_dir = "_test_generated_entries_exporter_timestamp"
    # Clean up previous test directory if it exists
    if os.path.exists(test_output_dir):
        import shutil
        shutil.rmtree(test_output_dir)
        print(f"Cleaned up old test directory: {test_output_dir}")
    # os.makedirs(test_output_dir) # Exporter __init__ will create it

    exporter = JournalExporter(test_output_dir)
    # today_date_str = datetime.now().strftime("%Y%m%d") # No longer needed for exporter direct call

    print(f"\nSaving entry 1...")
    entry1_path = exporter.save_entry("Test entry with timestamp ID 1.")
    if entry1_path: print(f"Saved: {entry1_path}")

    # Introduce a small delay to ensure the next timestamp is different if run very fast
    import time
    time.sleep(0.001) 

    print(f"\nSaving entry 2...")
    entry2_path = exporter.save_entry("Test entry with timestamp ID 2.")
    if entry2_path: print(f"Saved: {entry2_path}")

    print(f"\nCheck the directory '{os.path.abspath(test_output_dir)}' for the generated files (e.g., journal_YYYYMMDD_HHMMSSffffff.txt).") 