# Synthetic Journal Entry Generator

## Project Overview

This project generates synthetic daily journal entries using a Large Language Model (LLM). The primary goal is to create plausible and stylistically consistent journal data that can be used for downstream tasks, such as training tonal-analysis models. It leverages the Google Gemini API (specifically `gemini-1.5-flash-latest`) for text generation.

## Features

*   **LLM-Powered Generation:** Uses Google's Gemini 1.5 Flash model via API to generate journal entries.
*   **Configurable Output:** Allows users to specify the number of days, entries per day, average word count, and desired emotional tone.
*   **Emotion-Driven Prompts:** Can use examples from a seed dataset (Journal Entries with Labelled Emotions from Kaggle) for few-shot prompting to guide the LLM towards a specific emotional style.
*   **Unique File Naming:** Saves each entry as an individual `.txt` file with a unique timestamp-based name (e.g., `journal_YYYYMMDD_HHMMSSffffff.txt`) to prevent overwrites and ensure traceability.
*   **Text Processing:** Includes utilities for basic text cleaning and smart truncation to adhere to word count targets.
*   **Command-Line Interface:** Easy to run and configure via CLI arguments.
*   **Modular Structure:** Code is organized into modules for data loading, generation, exporting, and utilities.
*   **Unit Tests:** Includes a suite of unit tests using `pytest` to ensure code quality and correctness.

## Directory Structure

```
journal_generator/
├── .env                  # (User-created) For API keys and environment variables
├── .gitignore            # Specifies intentionally untracked files that Git should ignore
├── data/
│   ├── data.csv          # (User-provided) Seed data for journal entries and emotions
│   └── raw/              # (Optional) Can be used for other raw data assets
├── generated_entries/    # Default output directory for generated .txt journal files
├── src/                  # Source code
│   ├── __init__.py
│   ├── data_loader.py    # Loads and preprocesses seed data
│   ├── exporter.py       # Handles saving generated entries to files
│   ├── generator.py      # Core LLM interaction and prompt construction
│   ├── main.py           # Main script CLI and orchestration
│   └── utils.py          # Helper functions (text processing, file naming, etc.)
├── tests/                # Unit tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_exporter.py
│   ├── test_generator.py
│   └── test_utils.py
├── requirements.txt      # Python package dependencies
└── README.md             # This file
```

## Setup Instructions

Follow these steps to set up and run the journal generator:

**1. Python Version:**
   Ensure you have Python 3.10 or newer installed. You can check your Python version by running:
   ```bash
   python --version
   # or
   python3 --version
   ```

**2. Create and Activate a Virtual Environment:**
   It's highly recommended to use a virtual environment to manage project dependencies.

   Navigate to the project root directory (`journal_generator/`) in your terminal and run:
   ```bash
   # Create a virtual environment named 'venv'
   python3 -m venv venv

   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # .\venv\Scripts\activate 
   ```
   Your terminal prompt should now indicate that the `(venv)` is active.

**3. Install Dependencies:**
   With the virtual environment activated, install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

**4. Download NLTK Resources (One-time setup for `nltk.word_tokenize`):**
   The project uses NLTK for word counting and truncation. If you haven't used NLTK for tokenization before, you might need to download the `punkt` resource. Run the following in a Python interpreter (can be done from the activated venv):
   ```python
   import nltk
   nltk.download('punkt')
   ```
   You only need to do this once per Python environment.

**5. Set up Google API Key:**
   The generator uses the Google Gemini API, which requires an API key.
   *   Obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
   *   In the root directory of the project (`journal_generator/`), duplicate the `.env.example` file and name it `.env`.
   *   Add your API key to the `.env` file in the following format:
     ```
     GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
     ```
   *   **Important:** The `.env` file is listed in `.gitignore`, so your API key will not be committed to version control.

**6. (Optional) Seed Data for Few-Shot Prompting:**
   If you want to use the few-shot prompting feature (highly recommended for better tone control), download the "Journal Entries with Labelled Emotions" dataset from Kaggle:
   *   [madhavmalhotra/journal-entries-with-labelled-emotions](https://www.kaggle.com/datasets/madhavmalhotra/journal-entries-with-labelled-emotions)
   *   Place the `data.csv` file into the `journal_generator/data/` directory.
   *   If you do not provide this file or set `--num_examples_prompt 0`, the generator will still work but without dataset-specific examples in prompts.

## How it Works

1.  **Initialization (`src/main.py`):**
    *   Parses command-line arguments.
    *   Loads environment variables (including the `GOOGLE_API_KEY`) using `python-dotenv`.
    *   Optionally loads and preprocesses the seed data from `data/data.csv` using `src/data_loader.py` if few-shot examples are requested.
    *   Initializes the `JournalGenerator` (`src/generator.py`) which configures the Gemini API client.
    *   Initializes the `JournalExporter` (`src/exporter.py`) for saving entries.

2.  **Entry Generation Loop (`src/main.py`):**
    *   Iterates for the specified number of days and entries per day.
    *   For each entry:
        *   If examples are enabled, `src/data_loader.py` fetches relevant examples for the target emotion.
        *   `src/generator.py` constructs a detailed prompt including the target emotion, desired word count, and few-shot examples (if any).
        *   The prompt is sent to the Google Gemini API (`gemini-1.5-flash-latest`).
        *   The API's response (generated text) is received.
        *   `src/utils.py` functions are used to clean the text (`clean_generated_text`) and check word count adherence (`check_word_count_adherence`, `smart_truncate_text`).

3.  **Saving Entries (`src/exporter.py`):**
    *   Each processed journal entry is passed to the `JournalExporter`.
    *   `src/utils.py` generates a unique filename using the current timestamp: `get_current_datetime_str_for_file_id()` produces a `YYYYMMDD_HHMMSSffffff` string.
    *   `construct_filename()` creates the final name, e.g., `journal_YYYYMMDD_HHMMSSffffff.txt`.
    *   The entry is saved as a `.txt` file in the specified output directory (default: `generated_entries/`).

## File Naming Convention

Generated journal entries are saved in the directory specified by `--output_dir` (defaults to `generated_entries/`).
The naming convention for each file is:
`journal_YYYYMMDD_HHMMSSffffff.txt`

Where:
*   `YYYYMMDD`: Year, Month, Day
*   `HHMMSS`: Hour, Minute, Second
*   `ffffff`: Microseconds

This ensures that each generated file has a unique name.

## Running the Script

Activate your virtual environment first (`source venv/bin/activate`). Then, run the main script from the project root directory:

```bash
python src/main.py --tone <emotion> [OPTIONS]
```

**Required Argument:**

*   `--tone <emotion>`: Desired tone/emotion for the entries.
    *   This argument is **required**.
    *   The chosen emotion **must be one of the 18 specific emotions** available in the seed dataset (if used for examples) and supported by the script. These are used to guide the LLM.
    *   Available choices: `accomplished`, `afraid`, `anxious`, `disappointed`, `excited`, `frustrated`, `grateful`, `happy`, `inspired`, `lonely`, `love`, `motivated`, `nostalgic`, `proud`, `reflective`, `sad`, `stressed`, `surprised`.
    *   Example: `python src/main.py --tone "happy"`

**Optional Arguments:**

*   `--num_days <int>`: Number of days to generate entries for (default: 1).
*   `--entries_per_day <int>`: Number of entries to generate per day (default: 1).
*   `--avg_word_count <int>`: Desired average word count for entries (default: 100).
*   `--output_dir <path>`: Directory to save generated `.txt` files (default: `generated_entries`).
*   `--num_examples_prompt <int>`: Number of example entries from the dataset to use in the few-shot prompt. Set to 0 to disable. (default: 3).
*   `--max_generation_tokens <int>`: Maximum number of new tokens the LLM should generate. Default 0 lets the generator estimate based on `avg_word_count`.
*   `--start_date <YYYYMMDD>`: Start date for journal entries in YYYYMMDD format. Defaults to the current day. Example: `--start_date 20240115`.

**Example Usage:**

Generate 2 entries per day for 3 days, with a "reflective" tone, aiming for 150 words each, using 2 examples for prompting, and saving to `my_journals/`:
```bash
python src/main.py --entries_per_day 3 --avg_word_count 200 --tone "awkward"
```

## Running Tests

The project uses `pytest` for unit testing.

**1. Install Pytest:**
   `pytest` is included in `requirements.txt`, so it should be installed when you set up your environment.

**2. Run Tests:**
   Navigate to the project root directory (`journal_generator/`) in your terminal (with the virtual environment activated) and run:
   ```bash
   pytest
   ```
   `pytest` will automatically discover and run all tests in the `tests/` directory. Ensure NLTK's `punkt` resource is downloaded as mentioned in the setup.

This should provide a solid foundation for understanding and using your project! 