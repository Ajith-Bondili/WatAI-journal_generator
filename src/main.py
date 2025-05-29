import argparse
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
# This should be done as early as possible, especially before other modules might try to access them.
load_dotenv() 

# Try to import from local src package first
if __package__:
    from .data_loader import load_and_preprocess_data, get_examples_for_prompt, ALL_AVAILABLE_EMOTIONS
    from .generator import JournalGenerator
    from .exporter import JournalExporter
    from .utils import get_current_datetime_str # For default start date
else:
    # Allow running directly from src/ for simplified testing, assuming other files are in the same dir
    from data_loader import load_and_preprocess_data, get_examples_for_prompt, ALL_AVAILABLE_EMOTIONS
    from generator import JournalGenerator
    from exporter import JournalExporter
    from utils import get_current_datetime_str

DEFAULT_OUTPUT_DIR = "generated_entries"

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic journal entries.")
    parser.add_argument(
        "--num_days", 
        type=int, 
        default=1, # Made optional, default to 1 day
        help="Number of days to generate entries for (default: 1)."
    )
    parser.add_argument(
        "--entries_per_day", 
        type=int, 
        default=1, 
        help="Number of entries to generate per day (default: 1)."
    )
    parser.add_argument(
        "--avg_word_count", 
        type=int, 
        default=100, 
        help="Desired average word count for entries (default: 100)."
    )
    parser.add_argument(
        "--tone", 
        type=str, 
        required=True, 
        choices=ALL_AVAILABLE_EMOTIONS, # Use emotions from data_loader
        help=f"Desired tone/emotion for the entries. Choose from: {', '.join(ALL_AVAILABLE_EMOTIONS)}"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=DEFAULT_OUTPUT_DIR, 
        help=f"Directory to save generated .txt files (default: {DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--num_examples_prompt", 
        type=int, 
        default=3, 
        help="Number of example entries from the dataset to use in the few-shot prompt. Set to 0 to disable. (default: 3)."
    )
    parser.add_argument(
        "--max_generation_tokens",
        type=int,
        default=0, # Default to 0, let generator calculate based on avg_word_count
        help="Maximum number of new tokens the LLM should generate. Default 0 lets the generator estimate based on avg_word_count."
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None, 
        help="Start date for journal entries in YYYYMMDD format. Defaults to today."
    )

    args = parser.parse_args()

    print("--- Journal Generation Script Initializing ---")
    print(f"Configuration:\n{args}")

    journal_df = None
    if args.num_examples_prompt > 0:
        print("\n--- Loading and Preprocessing Seed Data for Examples ---")
        try:
            journal_df = load_and_preprocess_data()
            print("Seed data loaded successfully.")
        except Exception as e:
            print(f"Error loading seed data: {e}. Few-shot prompting with dataset examples will be disabled.")
            journal_df = None 
    else:
        print("\n--- Skipping seed data loading as num_examples_prompt is 0. ---")

    print("\n--- Initializing Modules ---")
    try:
        generator = JournalGenerator()
        # Ensure Exporter class name matches what's in exporter.py (e.g., Exporter or JournalExporter)
        exporter = JournalExporter(output_dir=args.output_dir) 
        print("Generator and Exporter initialized.")
    except Exception as e:
        print(f"Error initializing modules: {e}. Exiting.")
        return

    print("\n--- Starting Journal Entry Generation ---")
    total_entries_generated = 0
    current_date_obj = datetime.strptime(args.start_date, "%Y%m%d") if args.start_date else datetime.now()

    for day_num in range(args.num_days):
        date_str = (current_date_obj + timedelta(days=day_num)).strftime("%Y%m%d")
        print(f"\n== Day {day_num + 1} of {args.num_days} (Date: {date_str}) ==")
        
        for entry_num_in_day in range(1, args.entries_per_day + 1):
            print(f"-- Generating entry {entry_num_in_day} of {args.entries_per_day} for date {date_str}, tone: '{args.tone}' --")
            
            example_entries_for_prompt = [] # Initialize for each entry
            if journal_df is not None and args.num_examples_prompt > 0:
                # Fetch new examples for EACH entry to promote diversity
                print(f"Fetching {args.num_examples_prompt} examples for tone: '{args.tone}'")
                example_entries_for_prompt = get_examples_for_prompt(journal_df, args.tone, args.num_examples_prompt)
                if not example_entries_for_prompt:
                    print(f"Warning: Could not fetch examples for tone '{args.tone}'. Proceeding without few-shot examples for this specific entry.")
            
            start_time = time.time()
            generated_text = generator.generate_entry(
                target_emotion=args.tone,
                avg_word_count=args.avg_word_count,
                example_entries=example_entries_for_prompt, # Pass potentially new examples
                max_new_tokens=args.max_generation_tokens
            )
            end_time = time.time()
            print(f"LLM Generation time: {end_time - start_time:.2f} seconds.")

            if generated_text:
                saved_file = exporter.save_entry(
                    entry_text=generated_text,
                    date_str=date_str,
                    entry_index_in_day=entry_num_in_day,
                    total_entries_for_day=args.entries_per_day
                )
                if saved_file:
                    total_entries_generated += 1
            else:
                print(f"Failed to generate entry {entry_num_in_day} for day {day_num + 1}. Skipping.")

    print("\n--- Journal Generation Complete ---")
    print(f"Total entries generated: {total_entries_generated}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    main() 