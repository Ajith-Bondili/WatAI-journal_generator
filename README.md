# Synthetic Journal Entry Generator

## Project Overview

This project generates synthetic daily journal entries using a Large Language Model (LLM). The primary goal is to create plausible and stylistically consistent journal data that can be used for downstream tasks, such as training tonal-analysis models.

## Features

*   **LLM-Powered Generation:** Uses Google's Gemini 1.5 Flash model via API to generate journal entries as its cheap and relatively low latency
*   **Configurable Output:** Allows users to specify the number of days, entries per day, average word count, and desired emotional tone as well as many other parameters
*   **Emotion-Driven Prompts:** Can use examples from a seed dataset (Journal Entries with Labelled Emotions from Kaggle) for few-shot prompting to guide the LLM towards a specific emotional style.
*   **Unique File Naming:** Saves each entry as an individual `.txt` file with a unique timestamp-based name (e.g., `journal_YYYYMMDD_HHMMSSffffff.txt`) to prevent overwrites and ensure traceability.
*   **Text Processing:** Includes utilities for basic text cleaning and smart truncation to adhere to word count targets.
*   **Command-Line Interface:** Easy to run and configure via CLI arguments, as well the CLI displays the what's going on in the pipeline and useful metadata.
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
   The project includes a dataset for few-shot prompting (highly recommended for better tone control) in the `journal_generator/data/` directory. This dataset was sourced from [madhavmalhotra/journal-entries-with-labelled-emotions](https://www.kaggle.com/datasets/madhavmalhotra/journal-entries-with-labelled-emotions) on Kaggle.
   *   The included `data.csv` file contains journal entries labeled with one or more of the following emotions: `accomplished`, `afraid`, `anxious`, `disappointed`, `excited`, `frustrated`, `grateful`, `happy`, `inspired`, `lonely`, `love`, `motivated`, `nostalgic`, `proud`, `reflective`, `sad`, `stressed`, `surprised`.
   *   If you set `--num_examples_prompt 0`, the generator will still work but without dataset-specific examples in prompts.

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

After activating your virtual environment (`source venv/bin/activate`) and installing the requirements.txt (`pip install -r requirments.txt`). Then, run the main script from the project root directory:

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

**Test File Overview:**

*   `tests/test_data_loader.py`: Verifies loading and preprocessing of seed data, and correct retrieval of emotional example entries.
*   `tests/test_exporter.py`: Ensures generated journal entries are correctly saved to files with proper naming and directory structure.
*   `tests/test_generator.py`: Checks the core journal entry generation logic, including prompt creation and (mocked) LLM API interactions.
*   `tests/test_main.py`: Tests the command-line interface, argument parsing, and the main script's orchestration of the generation process.
*   `tests/test_utils.py`: Validates various helper functions for text manipulation (cleaning, truncation) and utility tasks (like file naming).

This should provide a solid foundation for understanding and using your project! 

## Developer Notes & Reflections

This section includes some thoughts and challenges encountered during the development of this journal generator.

**Unique ID Generation:**
Initially, I considered a simple global counter for unique IDs for the journal entries. However, I quickly realized this approach could lead to ID collisions if the script were run multiple times, as the counter would reset. The current solution uses a timestamp-based approach (`YYYYMMDD_HHMMSSffffff`) to generate unique filenames, which is far more robust for preventing overwrites.

**LLM Choices & Local vs. API:**
The journey with LLMs involved a few iterations:
*   I started by looking into smaller, locally runnable models like Google's Flan-T5.
*   Then, I explored popular models on Hugging Face, such as `sarvamai/sarvam-m` and later considered options like `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.
*   The large size of more powerful models made local execution challenging and prompted thoughts of migrating to Google Colab for more GPU resources.
*   Ultimately, for this project, I opted to use the Google Gemini API (specifically `gemini-1.5-flash-latest`). This proved to be a practical choice due to its ease of use, cost-effectiveness (with a generous free tier that's unlikely to be exhausted for typical use cases of this script), and strong generative capabilities without the overhead of local model management.

**Controlling Average Word Count:**
Getting an LLM to adhere strictly to an "average word count" is trickier than it sounds. LLMs process text in terms of *tokens*, not words, and don't have an innate concept of word counts in the same way humans do.
*   Simply instructing the LLM to generate a certain number of words often yields approximate results.
*   More advanced solutions could involve creating a multi-step "agent" where one part of the LLM generates text and another refines it for length, or using a reasoning model to "think" about the length. However, these approaches can significantly increase computational expense and API costs.
*   The current solution takes a pragmatic approach: the prompt requests an *approximate* word count, and a post-generation utility function (`smart_truncate_text` in `utils.py`) is used. If the generated text's word count exceeds the target by more than a certain tolerance (e.g., 20%), it's truncated. If it's within a reasonable range (e.g., up to 20% over), it's considered acceptable. This balances output quality with simplicity and cost.
