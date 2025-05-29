import os
from datetime import datetime
import json # For storing the counter in a more structured way

# Try to import from local src package first, then parent directory if running script directly
if __package__:
    from . import utils
else:
    import utils # For standalone testing if necessary

class JournalExporter:
    # Path to the file that stores the global counter
    # Place it in the data directory to keep project root cleaner.
    # Ensure PROJECT_ROOT logic is sound if data_loader uses it, or define explicitly here.
    # For simplicity, assuming data_loader.py defines PROJECT_ROOT if needed, or this script is run
    # from a context where ../data is valid for the counter file.
    # Let's make it relative to this file for now, then to data/
    _COUNTER_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "_exporter_state.json")

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

        self.global_counter = self._load_counter()
        print(f"Exporter initialized. Output will be saved to: {os.path.abspath(self.output_dir)}. Initial global counter: {self.global_counter}")

    def _load_counter(self) -> int:
        """Loads the global counter from the state file."""
        try:
            if os.path.exists(self._COUNTER_FILE_PATH):
                with open(self._COUNTER_FILE_PATH, "r") as f:
                    data = json.load(f)
                    return int(data.get("global_counter", 0)) # Default to 0 if key missing
            return 0 # Default if file doesn't exist
        except (IOError, json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not load or parse counter file {self._COUNTER_FILE_PATH}: {e}. Starting counter at 0.")
            return 0

    def _save_counter(self):
        """Saves the current global counter to the state file."""
        try:
            # Ensure the data directory exists for the counter file
            os.makedirs(os.path.dirname(self._COUNTER_FILE_PATH), exist_ok=True)
            with open(self._COUNTER_FILE_PATH, "w") as f:
                json.dump({"global_counter": self.global_counter}, f)
        except IOError as e:
            print(f"Error: Could not save counter file {self._COUNTER_FILE_PATH}: {e}")

    def save_entry(
        self,
        entry_text: str,
        date_str: str, # e.g., "20231027"
        entry_index_in_day: int, # 1-based index for the entry of that day
        total_entries_for_day: int
    ) -> str | None:
        """
        Saves a single journal entry to a .txt file.

        Args:
            entry_text (str): The content of the journal entry.
            date_str (str): The date string for the entry (YYYYMMDD).
            entry_index_in_day (int): The 1-based index of this entry for the given day.
            total_entries_for_day (int): Total number of entries planned for that day (for filename padding).

        Returns:
            str | None: The full path to the saved file if successful, None otherwise.
        """
        if not entry_text:
            print("Warning: Attempted to save an empty entry. Skipping.")
            return None

        self.global_counter += 1 # Increment for each new entry
        filename = utils.construct_filename(date_str, self.global_counter)
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(entry_text)
            
            self._save_counter() # Save the new counter state after successful write
            print(f"Journal entry saved to: {filepath}")
            return filepath
        except IOError as e:
            print(f"Error saving journal entry to {filename}: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during save_entry: {e}")
            return None

if __name__ == '__main__':
    print("Testing JournalExporter with Global Counter...")
    test_output_dir = "_test_generated_entries_exporter"
    # Clean up previous test directory if it exists
    if os.path.exists(test_output_dir):
        import shutil
        shutil.rmtree(test_output_dir)
        print(f"Cleaned up old test directory: {test_output_dir}")
    os.makedirs(test_output_dir)

    # Ensure the data directory exists for the counter file for standalone test
    test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(test_data_dir, exist_ok=True)
    # For standalone testing, temporarily override the counter file path to be relative to this test setup
    # or ensure the default path will resolve correctly from where this test is run.
    # The default _COUNTER_FILE_PATH should work if src/exporter.py is run from the project root.
    # Let's ensure the counter file is reset for this test sequence.
    if os.path.exists(JournalExporter._COUNTER_FILE_PATH):
        os.remove(JournalExporter._COUNTER_FILE_PATH)
        print(f"Removed old counter file for test: {JournalExporter._COUNTER_FILE_PATH}")

    exporter = JournalExporter(test_output_dir)
    today_date_str = datetime.now().strftime("%Y%m%d")

    print(f"\nSaving entry 1 (global ID should be 1)... Current counter from exporter: {exporter.global_counter}")
    entry1_path = exporter.save_entry("Test entry for global ID 1.", today_date_str, 1, 10)
    if entry1_path: print(f"Saved: {entry1_path}, New counter: {exporter.global_counter}")

    print(f"\nSaving entry 2 (global ID should be 2)... Current counter from exporter: {exporter.global_counter}")
    entry2_path = exporter.save_entry("Test entry for global ID 2.", today_date_str, 2, 10)
    if entry2_path: print(f"Saved: {entry2_path}, New counter: {exporter.global_counter}")

    # Simulate closing the app and reopening by creating a new exporter instance
    print("\nSimulating script restart: Creating new exporter instance...")
    exporter_new_instance = JournalExporter(test_output_dir)
    print(f"New exporter instance loaded counter: {exporter_new_instance.global_counter} (should be 2 from previous run)")

    print(f"\nSaving entry 3 (global ID should be 3)... Current counter from new exporter: {exporter_new_instance.global_counter}")
    entry3_path = exporter_new_instance.save_entry("Test entry for global ID 3 (new instance).", today_date_str, 3, 10)
    if entry3_path: print(f"Saved: {entry3_path}, New counter: {exporter_new_instance.global_counter}")

    print(f"\nCheck the directory '{os.path.abspath(test_output_dir)}' for the generated files (e.g., journal_{today_date_str}_0001.txt).")
    print(f"Check the counter file at '{JournalExporter._COUNTER_FILE_PATH}' (should contain {exporter_new_instance.global_counter}).") 