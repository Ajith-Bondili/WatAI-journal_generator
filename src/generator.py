import google.generativeai as genai
import os
import time # For potential retries


# Try to import from local src package first, then parent directory if running script directly
if __package__:
    from . import utils
else:
    import utils # For standalone testing if necessary

# Model name for Gemini API
MODEL_NAME = "gemini-1.5-flash-latest"

print(f"Using LLM Model via API: {MODEL_NAME}")

class JournalGenerator:
    def __init__(self):
        """
        Initializes the journal generator with the Google Gemini API.
        """
        print(f"Initializing generator with Google Gemini API model: {MODEL_NAME}")
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it before running the script.")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(MODEL_NAME)
            print(f"Gemini model {MODEL_NAME} initialized successfully.")
        except Exception as e:
            print(f"Error initializing Google Gemini API: {e}")
            raise RuntimeError(f"Could not initialize Gemini model {MODEL_NAME}") from e

    def _construct_prompt_text(self, target_emotion: str, avg_word_count: int, example_entries: list[str] | None = None) -> str:
        """
        Constructs the prompt for the Gemini API.
        """
        prompt_parts = []
        prompt_parts.append(f"Write a journal entry that very much focuses on the tone/style preset: {target_emotion}.")
        prompt_parts.append(f"The entry must be approximately {avg_word_count} words long.")
        prompt_parts.append("Generate only the journal entry text itself, without any introductory phrases like 'Here is a journal entry:' or similar. Do not include any titles or extra formatting beyond standard paragraph breaks if needed.")

        if example_entries:
            prompt_parts.append("\nHere are some examples of style and tone to guide you:")
            for i, ex in enumerate(example_entries):
                # Basic cleaning, though Gemini is generally robust
                temp_ex = ex.replace('"""', '"').replace("'''", "'")
                # No complex escaping needed for Gemini prompt parts like for f-strings in local models
                prompt_parts.append(f'\nExample {i+1}: "{temp_ex}"') 
            prompt_parts.append("\nBased on these examples, and keeping a similar style and tone, write the new journal entry:")
        else:
            prompt_parts.append("\nWrite the new journal entry now, following all rules above:")
        
        return " ".join(prompt_parts)

    def generate_entry(
        self,
        target_emotion: str,
        avg_word_count: int,
        example_entries: list[str] | None = None,
        max_new_tokens: int = 0 # Note: Gemini uses max_output_tokens, we'll set generation_config
    ) -> str:
        """
        Generates a synthetic journal entry using the Gemini API.
        (Docstring for args remains similar)
        """
        if not self.model:
            print("Gemini model not initialized. Cannot generate entry.")
            return ""

        prompt_text = self._construct_prompt_text(target_emotion, avg_word_count, example_entries)
        
        # Gemini uses generation_config for parameters like max_output_tokens, temperature, etc.
        # Estimate max_output_tokens. For Gemini, token count is often similar to word count or slightly more.
        # Let's use a slightly more generous factor for safety, as max_output_tokens is a hard limit.
        if max_new_tokens == 0: # User didn't override
            # A heuristic: avg_word_count * 1.5 for tokens should be ample for Gemini Flash
            # plus a small buffer. Ensure a minimum if avg_word_count is very small.
            calculated_max_output_tokens = max(50, int(avg_word_count * 1.5) + 30)
        else: # User provided a value, assume it's intended for max_output_tokens
            calculated_max_output_tokens = max_new_tokens

        generation_config = genai.types.GenerationConfig(
            # candidate_count=1, # Default is 1
            # stop_sequences=[], # Can be used if needed
            max_output_tokens=calculated_max_output_tokens,
            temperature=0.7,  # Adjust as needed, 0.7 is a common default for creative tasks
            # top_p= N/A for gemini-1.5-flash directly, but often controlled by temperature
            # top_k= N/A for gemini-1.5-flash directly
        )
        # Safety settings can be adjusted if content is blocked too aggressively.
        # Default safety settings are generally good.
        # safety_settings = [
        #     { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" },
        #     # ... other categories ...
        # ]

        print("\n--- Sending PROMPT to Gemini API ---")
        print(prompt_text)
        print(f"(Generation Config: max_output_tokens={calculated_max_output_tokens}, temperature=0.7)")
        print("-----------------------------------")

        generated_text_raw = ""
        try:
            # For Gemini, the prompt is passed directly
            response = self.model.generate_content(
                prompt_text,
                generation_config=generation_config,
                # safety_settings=safety_settings 
            )
            
            print("\n--- Received RESPONSE from Gemini API ---")
            # print(response) # Full response object can be verbose
            
            # Accessing the text part:
            if response.parts:
                generated_text_raw = response.text # .text directly gives the combined text
            elif response.candidates and response.candidates[0].content.parts:
                 generated_text_raw = "".join(part.text for part in response.candidates[0].content.parts)
            else:
                # Fallback or if there was an issue like a block without explicit error parts
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    print(f"Warning: Prompt blocked by API. Reason: {response.prompt_feedback.block_reason}")
                else:
                    print(f"Warning: Unexpected Gemini API response format or empty response. Parts: {hasattr(response, 'parts')}, Candidates: {hasattr(response, 'candidates')}")
                # print(f"Full response for debugging: {response}") # uncomment for deep debug
                return ""

        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            # Consider adding retries with backoff for transient network issues if this becomes common
            return ""

        # Gemini usually doesn't include the prompt or extra preambles if instructed not to.
        # The clean_generated_text might still be useful for other artifacts.
        cleaned_text = utils.clean_generated_text(generated_text_raw) 
        current_word_count = utils.count_words(cleaned_text)

        print(f"Generated entry (raw, cleaned): {current_word_count} words.")
        print(f"Target word count: {avg_word_count}.")

        is_adherent, deviation = utils.check_word_count_adherence(current_word_count, avg_word_count)

        if is_adherent:
            print(f"Word count ({current_word_count}) is within +/-20% tolerance of target ({avg_word_count}). Deviation: {deviation:.2%}")
            final_text = cleaned_text
        else:
            print(f"Word count ({current_word_count}) is outside +/-20% tolerance of target ({avg_word_count}). Deviation: {deviation:.2%}. Attempting to adjust.")
            if current_word_count > avg_word_count: 
                print("Text is too long, applying smart truncation.")
                final_text = utils.smart_truncate_text(cleaned_text, avg_word_count, max_overshoot_words=int(avg_word_count*0.10))
            else: 
                print("Text is too short. Using as is. Consider increasing max_output_tokens or re-prompting (not implemented).")
                final_text = cleaned_text 
        
        final_word_count = utils.count_words(final_text)
        print(f"\n--- FINAL Processed Entry ({final_word_count} words) ---")
        print(final_text)
        print("---------------------------")
        return final_text

# Example of how to test this generator (if run directly)
if __name__ == '__main__':
    print(f"Testing JournalGenerator with Google Gemini API ({MODEL_NAME})...")
    print("Ensure GOOGLE_API_KEY environment variable is set.")
    
    mock_example_entries_reflective = [
        "I spent the evening looking at old photos. So many memories, some happy, some bittersweet. It made me think about how much things have changed.",
        "Quiet day today. I read a book by the window and just watched the world go by. It's good to have these moments of calm reflection."
    ]

    try:
        generator = JournalGenerator() 

        print(f"\n--- Test 1: Generating a REFLECTIVE entry with examples ({MODEL_NAME}) ---")
        reflective_entry = generator.generate_entry(
            target_emotion="reflective", 
            avg_word_count=60, 
            example_entries=mock_example_entries_reflective,
            max_new_tokens=120 # This will be used as max_output_tokens for Gemini
        )
        if reflective_entry:
            print(f"Generated Reflective Entry (approx 60 words):\n{reflective_entry}")
        else:
            print("Failed to generate reflective entry.")

        print(f"\n--- Test 2: Generating an ANXIOUS entry without examples ({MODEL_NAME}) ---")
        anxious_entry = generator.generate_entry(
            target_emotion="anxious", 
            avg_word_count=40,
            max_new_tokens=80
        )
        if anxious_entry:
            print(f"Generated Anxious Entry (approx 40 words):\n{anxious_entry}")
        else:
            print("Failed to generate anxious entry.")

        # Test with data_loader (if utils.py and data_loader.py are accessible)
        try:
            # This assumes data_loader.py is in the same directory or src is in PYTHONPATH
            # and that GOOGLE_API_KEY is set for the generator to initialize.
            from data_loader import load_and_preprocess_data, get_examples_for_prompt, ALL_AVAILABLE_EMOTIONS
            
            print(f"\n--- Test 3: Generating with actual data ({MODEL_NAME}) ---")
            # Ensure data_loader.py uses a path accessible from where you run this test
            # For local testing, ensure data/data.csv exists relative to data_loader.py's assumptions
            df = load_and_preprocess_data()
            if not df.empty and ALL_AVAILABLE_EMOTIONS:
                target_emotion_from_data = ALL_AVAILABLE_EMOTIONS[0] 
                print(f"Using emotion: {target_emotion_from_data}")
                examples = get_examples_for_prompt(df, target_emotion_from_data, 2)
                
                if examples:
                    print(f"Found {len(examples)} examples for '{target_emotion_from_data}'")
                else:
                    print(f"No examples found for '{target_emotion_from_data}', proceeding without them.")

                entry_with_data = generator.generate_entry(
                    target_emotion=target_emotion_from_data,
                    avg_word_count=50,
                    example_entries=examples,
                    max_new_tokens=100
                )
                if entry_with_data:
                    print(f"Generated '{target_emotion_from_data}' Entry (approx 50 words):\n{entry_with_data}")
                else:
                    print(f"Failed to generate '{target_emotion_from_data}' entry with data.")
            else:
                print("Could not load data or no emotions available for Test 3.")

        except ImportError as ie:
            print(f"Could not import data_loader for Test 3: {ie}. Ensure it's accessible.")
        except Exception as e_dl:
            print(f"Could not run data_loader dependent test (Test 3): {e_dl}")

    except RuntimeError as e:
        print(f"Could not initialize JournalGenerator: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during generator test: {e}") 