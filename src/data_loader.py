import pandas as pd
import nltk
import random
import os # Added for path joining

# Ensure NLTK's sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Path to the local data file
# Assuming the script is run from a context where this relative path is valid
# (e.g., from the project root, and src is in PYTHONPATH)
# Or, construct a path relative to this file's location.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # This goes up from src to journal_generator
LOCAL_DATA_FILE = os.path.join(PROJECT_ROOT, "data", "data.csv") # User placed it in data/data.csv

EMOTION_COLUMNS = [
    'Answer.f1.afraid.raw',
    'Answer.f1.angry.raw',
    'Answer.f1.anxious.raw',
    'Answer.f1.ashamed.raw',
    'Answer.f1.awkward.raw',
    'Answer.f1.bored.raw',
    'Answer.f1.calm.raw',
    'Answer.f1.confused.raw',
    'Answer.f1.disgusted.raw',
    'Answer.f1.excited.raw',
    'Answer.f1.frustrated.raw',
    'Answer.f1.happy.raw',
    'Answer.f1.jealous.raw',
    'Answer.f1.nostalgic.raw',
    'Answer.f1.proud.raw',
    'Answer.f1.sad.raw',
    'Answer.f1.satisfied.raw',
    'Answer.f1.surprised.raw'
]

ALL_AVAILABLE_EMOTIONS = [col.split('.')[2] for col in EMOTION_COLUMNS]

def load_and_preprocess_data() -> pd.DataFrame:
    """
    Loads the dataset from the local CSV file and performs basic preprocessing.
    Converts emotion columns from TRUE/FALSE strings to booleans.

    Returns:
        pd.DataFrame: The loaded and preprocessed DataFrame.

    Raises:
        FileNotFoundError: If the local dataset CSV cannot be loaded.
        KeyError: If essential columns like 'Answer' or emotion columns are missing.
    """
    data_file_path = LOCAL_DATA_FILE
    print(f"Attempting to load local dataset: {data_file_path}")
    
    if not os.path.exists(data_file_path):
        print(f"Error: Local data file not found at {data_file_path}")
        print(f"Please ensure 'data.csv' is placed in the '{os.path.join(PROJECT_ROOT, "data")}' directory.")
        raise FileNotFoundError(f"Could not load {data_file_path}. File does not exist.")

    try:
        df = pd.read_csv(data_file_path)
        print(f"Local dataset loaded successfully. Shape: {df.shape}")

    except Exception as e:
        print(f"Error loading local dataset from {data_file_path}: {e}")
        raise FileNotFoundError(f"Could not load {data_file_path}.") from e

    if 'Answer' not in df.columns:
        raise KeyError("Critical column 'Answer' not found in the dataset.")

    # Preprocess emotion columns: Convert string 'TRUE'/'FALSE' to boolean
    for col in EMOTION_COLUMNS:
        if col in df.columns:
            if df[col].dtype == 'object': # Check if column is object type (likely strings)
                # Handle potential mixed case like 'True' or 'False' and also NaN before mapping
                df[col] = df[col].astype(str).str.upper().map({'TRUE': True, 'FALSE': False, 'NAN': False}).fillna(False)
            elif pd.api.types.is_numeric_dtype(df[col]): # If it's numeric (0/1)
                 df[col] = df[col].astype(bool)
            # If already boolean, no change needed
        else:
            print(f"Warning: Emotion column {col} not found in the dataset. It will be ignored.")

    df['Answer'] = df['Answer'].astype(str).str.strip()
    return df

def get_examples_for_prompt(df: pd.DataFrame, target_emotion: str, num_examples: int = 3) -> list[str]:
    """
    Retrieves a specified number of example journal entries for a given target emotion.

    Args:
        df (pd.DataFrame): The DataFrame containing journal entries and emotion labels.
        target_emotion (str): The desired emotion (e.g., 'Happy', 'Sad'). Case-insensitive.
        num_examples (int): The number of example entries to retrieve.

    Returns:
        list[str]: A list of example journal entry texts. Returns empty list if no matches or errors.
    """
    if num_examples <= 0:
        return []

    target_emotion_col_name = f"Answer.f1.{target_emotion.lower()}.raw"

    if target_emotion_col_name not in df.columns:
        print(f"Warning: Emotion column for '{target_emotion}' ({target_emotion_col_name}) not found. Cannot fetch examples.")
        return []

    if not pd.api.types.is_bool_dtype(df[target_emotion_col_name]):
        print(f"Warning: Emotion column {target_emotion_col_name} is not boolean. Cannot reliably filter.")
        if pd.api.types.is_numeric_dtype(df[target_emotion_col_name]):
            df[target_emotion_col_name] = df[target_emotion_col_name].astype(bool)
        else:
            return [] 

    filtered_df = df[df[target_emotion_col_name] == True]

    if filtered_df.empty:
        print(f"No entries found for emotion: {target_emotion}")
        return []

    if len(filtered_df) < num_examples:
        print(f"Warning: Found only {len(filtered_df)} entries for emotion '{target_emotion}', requested {num_examples}. Using all found.")
        return filtered_df['Answer'].tolist()
    
    return random.sample(filtered_df['Answer'].tolist(), num_examples)

if __name__ == '__main__':
    try:
        print(f"Looking for data file at: {LOCAL_DATA_FILE}")
        if not os.path.exists(LOCAL_DATA_FILE):
            print(f"STANDALONE TEST ERROR: {LOCAL_DATA_FILE} not found.")
            print(f"Make sure you have downloaded data.csv and placed it in {os.path.join(PROJECT_ROOT, 'data')}")
        else:
            journal_df = load_and_preprocess_data()
            print("\n--- Data Schema after Preprocessing ---")
            journal_df.info()
            print("\n--- Example Rows (First 5) ---")
            print(journal_df.head())

            print(f"\n--- Available Emotions for Prompting ---")
            print(ALL_AVAILABLE_EMOTIONS)

            emotion_to_test = 'happy' 
            num_exp = 2
            print(f"\n--- Fetching {num_exp} examples for emotion: '{emotion_to_test}' ---")
            examples = get_examples_for_prompt(journal_df, emotion_to_test, num_exp)
            if examples:
                for i, example in enumerate(examples):
                    print(f"Example {i+1}:\n{example[:200]}...\n")
            else:
                print(f"No examples found for '{emotion_to_test}'.")

    except (FileNotFoundError, KeyError) as e:
        print(f"Failed to run data loader example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during data loader example: {e}") 