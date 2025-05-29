# Journal Generator

## 1. Objective
Build a module to automatically generate and export synthetic daily journal entries for use in downstream tonal-analysis model training.

## 2. Scope & Requirements Summary
- Ingest public daily-journal datasets (using Kaggle dataset: "madhavmalhotra/journal-entries-with-labelled-emotions").
- Expand the seed dataset with novel entries using an LLM (e.g., `sarvamai/sarvam-m`), preserving realistic style, tone, and length.
- Save each generated entry as an individual `.txt` file (e.g., `journal_YYYYMMDD_<id>.txt`).
- Configure parameters for:
    - Number of entries per day.
    - Desired average word count.
    - Tone/style presets (based on emotions from the dataset).
- Provide clear README instructions and unit tests.

## 3. Project Structure
```
journal_generator/
├── data/
│   └── raw/                # Optional: for manually placed data.csv
├── generated_entries/      # Output directory for .txt files
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Loads and preprocesses data
│   ├── generator.py        # Generates synthetic journal entries
│   ├── exporter.py         # Exports entries to .txt files
│   ├── main.py             # Main script to run the generator
│   └── utils.py            # Utility functions (date, IDs, etc.)
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_generator.py
│   └── test_exporter.py
├── README.md
└── requirements.txt
```

## 4. Setup

### 4.1. Prerequisites
- Python 3.8+
- Kaggle API Key: Ensure you have your Kaggle API key set up. This typically involves:
    1. Go to your Kaggle account settings page (`https://www.kaggle.com/<your-username>/account`).
    2. Click on "Create New API Token". This will download `kaggle.json`.
    3. Place `kaggle.json` in the correct directory:
        - Linux/macOS: `~/.kaggle/kaggle.json`
        - Windows: `C:\Users\<Windows-User>\.kaggle\kaggle.json`
    4. Set permissions for the file (on Linux/macOS): `chmod 600 ~/.kaggle/kaggle.json`

### 4.2. Installation
1.  Clone the repository (if applicable) or create the project directory.
2.  Navigate to the project directory: `cd journal_generator`
3.  Create a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5.  Download NLTK resources (one-time setup):
    ```python
    import nltk
    nltk.download('punkt')
    ```
    You can do this by running the Python interpreter and typing the commands above.

## 5. Usage
The main script `src/main.py` is used to generate journal entries.

### 5.1. Command-Line Arguments
- `--num_days`: (int, required) Number of days to generate entries for.
- `--entries_per_day`: (int, default: 1) Number of entries to generate per day.
- `--avg_word_count`: (int, default: 100) Desired average word count for entries.
- `--tone`: (str, required) Desired tone/emotion for the entries. Choose from:
    `Happy`, `Satisfied`, `Calm`, `Proud`, `Excited`, `Frustrated`, `Anxious`, `Surprised`, `Nostalgic`, `Bored`, `Sad`, `Angry`, `Confused`, `Disgusted`, `Afraid`, `Ashamed`, `Awkward`, `Jealous`.
    (These are based on the emotions in the dataset).
- `--output_dir`: (str, default: "generated_entries") Directory to save generated .txt files.
- `--num_examples_prompt`: (int, default: 3) Number of example entries from the dataset to use in the few-shot prompt. Set to 0 to disable few-shot examples.
- `--max_generation_tokens`: (int, default: 150) Max new tokens for the LLM. Adjust based on `avg_word_count`.

### 5.2. Example
```bash
python src/main.py --num_days 2 --entries_per_day 3 --avg_word_count 75 --tone "Reflective"
```
This will generate 3 entries per day for 2 days (total 6 entries), aiming for an average of 75 words each, with a "Reflective" tone, and save them in the `generated_entries` directory.

To generate entries with a "Happy" tone:
```bash
python src/main.py --num_days 1 --entries_per_day 5 --avg_word_count 50 --tone "Happy"
```

## 6. Running Tests
To run the unit tests:
```bash
python -m unittest discover -s tests
```
Ensure your `PYTHONPATH` is set correctly if running from the root project directory, or `cd` into `src` and adjust paths if needed, though `discover` should handle it from the root.
One common way:
```bash
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"
```

## 7. Data Analysis Script
A script `analyze_data.py` (provided separately or can be created by the user) can be used to analyze the characteristics of the source dataset:
```bash
# (Create analyze_data.py with the provided code)
# Ensure kaggle API is set up and dependencies are installed
# python analyze_data.py
```
This script will output statistics on word counts, sentence counts, and emotion/topic prevalence from the Kaggle dataset, which helps in choosing appropriate parameters for `main.py`. 