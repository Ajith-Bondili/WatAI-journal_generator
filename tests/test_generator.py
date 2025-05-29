import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import os
import sys

# Add the src directory to the Python path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "src")
sys.path.insert(0, src_dir)

# Now import after path setup
from generator import JournalGenerator, MODEL_NAME # Import MODEL_NAME for checks
import utils
import google.generativeai as genai # For type hinting the mock response

class TestJournalGenerator(unittest.TestCase):

    @patch('os.getenv') # Patch os.getenv first
    @patch('google.generativeai.GenerativeModel') # Then patch GenerativeModel
    def setUp(self, mock_generative_model, mock_os_getenv):
        """Set up for each test. Mock environment variables and Gemini model."""
        # Configure os.getenv mock
        mock_os_getenv.return_value = "FAKE_API_KEY" # Mock API key

        # Configure the mock for genai.GenerativeModel instance
        self.mock_model_instance = MagicMock()
        # The generate_content method needs to return an object that mimics Gemini's response.
        # This response object should have a .text attribute or .parts attribute.
        mock_gemini_response = MagicMock(spec=genai.types.GenerateContentResponse)
        mock_gemini_response.text = "This is a mock LLM response from Gemini." # Default mock response text
        # For more complex scenarios, you might mock response.parts or response.candidates
        # e.g. type(mock_gemini_response).parts = PropertyMock(return_value=[MagicMock(text=...)])
        self.mock_model_instance.generate_content.return_value = mock_gemini_response
        
        # Make genai.GenerativeModel return our instance mock
        mock_generative_model.return_value = self.mock_model_instance

        # Store mocks for assertions
        self.mock_os_getenv = mock_os_getenv
        self.mock_generative_model_class = mock_generative_model # The class mock

        # Now, when JournalGenerator is instantiated, it will use our mocks
        try:
            self.generator = JournalGenerator()
        except Exception as e:
            self.fail(f"JournalGenerator instantiation failed even with mocks: {e}")

    def tearDown(self):
        """Clean up after each test by stopping all patches if necessary."""
        # Patches started with @patch decorators are automatically stopped.
        # If we started any patches manually with .start(), we'd call .stop() here.
        pass

    def test_generator_initialization(self):
        """Test that the JournalGenerator initializes correctly with mocks."""
        self.assertIsNotNone(self.generator, "Generator should be initialized.")
        self.assertIsNotNone(self.generator.model, "Gemini model should be set on generator.")
        self.assertEqual(self.generator.model, self.mock_model_instance, "Generator model should be our mocked instance.")

        # Check that os.getenv was called for the API key
        self.mock_os_getenv.assert_any_call("GOOGLE_API_KEY")

        # Check that genai.GenerativeModel (the class) was called to create an instance
        self.mock_generative_model_class.assert_called_once_with(MODEL_NAME)

    def test_construct_prompt_text_basic(self):
        """Test the basic prompt text construction without examples."""
        target_emotion = "curious"
        avg_word_count = 50
        prompt_text = self.generator._construct_prompt_text(target_emotion, avg_word_count)
        
        self.assertIsInstance(prompt_text, str)
        self.assertIn(f"tone/style preset: {target_emotion}", prompt_text)
        self.assertIn(f"approximately {avg_word_count} words long", prompt_text)
        self.assertIn("Generate only the journal entry text itself", prompt_text)
        self.assertNotIn("Example 1:", prompt_text)
        self.assertIn("Write the new journal entry now", prompt_text)

    def test_construct_prompt_text_with_examples(self):
        """Test prompt text construction with few-shot examples."""
        target_emotion = "excited"
        avg_word_count = 70
        example_entries = ["Example entry 1 about excitement.", "Example entry 2, also very exciting!"]
        prompt_text = self.generator._construct_prompt_text(target_emotion, avg_word_count, example_entries)
        
        self.assertIsInstance(prompt_text, str)
        self.assertIn(f"tone/style preset: {target_emotion}", prompt_text)
        self.assertIn(f"approximately {avg_word_count} words long", prompt_text)
        self.assertIn("Example 1: \"Example entry 1 about excitement.\"", prompt_text)
        self.assertIn(f'Example 2: "{example_entries[1]}"', prompt_text) # Using f-string for the second example
        self.assertIn("Based on these examples", prompt_text)

    def test_generate_entry_calls_gemini_and_processes_response(self):
        """Test that generate_entry calls the Gemini API and processes the response."""
        target_emotion = "nostalgic"
        avg_word_count = 60
        max_tokens_override = 0 # Let generator calculate
        expected_mock_response_text = "A nostalgic piece from mock Gemini."

        # Configure the mock model instance's generate_content for this test
        mock_response = MagicMock(spec=genai.types.GenerateContentResponse)
        mock_response.text = expected_mock_response_text
        self.mock_model_instance.generate_content.return_value = mock_response

        generated_text = self.generator.generate_entry(
            target_emotion, 
            avg_word_count, 
            max_new_tokens=max_tokens_override
        )
        
        self.mock_model_instance.generate_content.assert_called_once() # Check it was called
        args_call, kwargs_call = self.mock_model_instance.generate_content.call_args
        
        # args_call[0] should be the prompt string
        self.assertIsInstance(args_call[0], str)
        self.assertIn(target_emotion, args_call[0])
        
        # kwargs_call should contain generation_config
        self.assertIn('generation_config', kwargs_call)
        gen_config = kwargs_call['generation_config']
        self.assertIsInstance(gen_config, genai.types.GenerationConfig)
        
        # Check some default generation_config values (or calculated ones)
        # Heuristic from generator: max(50, int(avg_word_count * 1.5) + 30)
        expected_max_output_tokens = max(50, int(avg_word_count * 1.5) + 30)
        self.assertEqual(gen_config.max_output_tokens, expected_max_output_tokens)
        self.assertEqual(gen_config.temperature, 0.7)
        
        # Assuming utils.clean_generated_text just strips, and no truncation needed for this length
        self.assertEqual(generated_text, expected_mock_response_text.strip()) 

    def test_generate_entry_handles_gemini_failure(self):
        """Test that generate_entry returns empty string if Gemini API call fails."""
        self.mock_model_instance.generate_content.side_effect = Exception("Gemini simulated error")
        generated_text = self.generator.generate_entry("any_emotion", 50)
        self.assertEqual(generated_text, "")

    def test_generate_entry_handles_unexpected_gemini_response(self):
        """Test handling of unexpected response format or blocked prompt from Gemini."""
        # Scenario 1: Response object with no .text and no .parts (or empty parts)
        mock_response_no_text = MagicMock(spec=genai.types.GenerateContentResponse)
        # Ensure .text would raise or be None, and .parts is not there or empty
        type(mock_response_no_text).text = PropertyMock(side_effect=AttributeError('no text attribute'))
        type(mock_response_no_text).parts = PropertyMock(return_value=[]) # No parts
        # Simulate a prompt block reason as well
        mock_response_no_text.prompt_feedback = MagicMock()
        mock_response_no_text.prompt_feedback.block_reason = "SAFETY"
        
        self.mock_model_instance.generate_content.return_value = mock_response_no_text
        generated_text = self.generator.generate_entry("any_emotion", 50)
        self.assertEqual(generated_text, "", "Should return empty string for unusable Gemini response with block reason.")

        # Scenario 2: Response object with .candidates but empty parts inside candidate
        mock_candidate_part = MagicMock()
        mock_candidate_part.text = ""
        mock_candidate = MagicMock()
        mock_candidate.content = MagicMock()
        mock_candidate.content.parts = [mock_candidate_part] # Empty text in part
        mock_response_empty_candidate_parts = MagicMock(spec=genai.types.GenerateContentResponse)
        type(mock_response_empty_candidate_parts).text = PropertyMock(side_effect=AttributeError('no direct text attribute'))
        type(mock_response_empty_candidate_parts).parts = PropertyMock(return_value=None) # No direct parts
        mock_response_empty_candidate_parts.candidates = [mock_candidate]
        mock_response_empty_candidate_parts.prompt_feedback = None # No block reason this time

        self.mock_model_instance.generate_content.return_value = mock_response_empty_candidate_parts
        generated_text = self.generator.generate_entry("another_emotion", 50)
        self.assertEqual(generated_text, "", "Should return empty string for Gemini response with empty candidate parts.")

    def test_generated_text_cleaning_and_truncation(self):
        """Test that generated text is cleaned and then potentially truncated by utils."""
        raw_llm_output = "  This is a mock response that is deliberately a bit too long for the target word count. It needs to be truncated.  "
        # Note: The old test had "New Journal Entry:" which was stripped by the generator. 
        # The new generator logic has the LLM omit this, so clean_generated_text primarily strips whitespace.
        
        target_word_count = 10 # Target for truncation
        # Heuristic from generator for max_overshoot_words: int(avg_word_count*0.10)
        expected_max_overshoot_words = int(target_word_count * 0.10) 

        # Configure the mock Gemini response for this test
        mock_response = MagicMock(spec=genai.types.GenerateContentResponse)
        mock_response.text = raw_llm_output
        self.mock_model_instance.generate_content.return_value = mock_response
        
        # We will patch the actual utils functions to assert they are called correctly
        # and to control their output for focused testing of the generator's logic.
        with patch.object(utils, 'clean_generated_text', return_value=raw_llm_output.strip()) as mock_clean:
            # Let smart_truncate_text run, but we can check its call if needed, or check its effect.
            # For this test, let's assume smart_truncate_text works as per its own unit tests,
            # and focus on the fact that generate_entry uses it correctly when text is too long.
            # To make the test more specific to the generator's branching logic for truncation:
            
            # Scenario 1: Text is too long and needs truncation
            # Make clean_generated_text return something that IS too long
            text_that_is_too_long_after_cleaning = "This is cleaned but still very very long and needs truncation for sure."
            # utils.count_words(text_that_is_too_long_after_cleaning) -> 12. Target is 10. 20% tolerance is 8-12. So this should be adherent.
            # Let's make it longer to force truncation: target 5, actual 12
            forced_target_word_count = 5
            expected_max_overshoot_for_forced = int(forced_target_word_count * 0.10) # which is 0
            mock_clean.return_value = text_that_is_too_long_after_cleaning # Output of clean
            
            # Expected output after smart_truncate_text if it were called with this input:
            # utils.smart_truncate_text(text_that_is_too_long_after_cleaning, forced_target_word_count, expected_max_overshoot_for_forced)
            # words = ["This", "is", "cleaned", "but", "still", "very", "very", "long", "and", "needs", "truncation", "for", "sure", "."]
            # truncated_words = words[:5] -> ["This", "is", "cleaned", "but", "still"]
            # expected_truncated_text = "This is cleaned but still"

            # Patch smart_truncate_text to control its output and check its args for this specific path
            with patch.object(utils, 'smart_truncate_text', return_value="Successfully Truncated Text.") as mock_smart_truncate:
                generated_text = self.generator.generate_entry(
                    target_emotion="test_truncation", 
                    avg_word_count=forced_target_word_count 
                    # max_new_tokens will be calculated by the generator based on avg_word_count
                )
                
                mock_clean.assert_called_with(raw_llm_output) # Called with raw LLM output
                # smart_truncate_text should be called because text_that_is_too_long_after_cleaning (12 words) 
                # is outside tolerance for forced_target_word_count (5 words). (5 +/- 20% is 4-6 words)
                mock_smart_truncate.assert_called_once_with(text_that_is_too_long_after_cleaning, forced_target_word_count, max_overshoot_words=expected_max_overshoot_for_forced)
                self.assertEqual(generated_text, "Successfully Truncated Text.")

        # Scenario 2: Text is within tolerance and should not be truncated
        self.mock_model_instance.generate_content.reset_mock() # Reset call count for generate_content
        # mock_clean.reset_mock() # No longer needed as we will re-patch
        
        text_that_is_good_length = "This text is a good length."
        # utils.count_words(text_that_is_good_length) -> 6. Target 6, 20% tolerance is ~5-7. Adherent.
        target_good_length = 6
        
        # Configure the mock Gemini response for Scenario 2
        mock_response_scenario2 = MagicMock(spec=genai.types.GenerateContentResponse)
        mock_response_scenario2.text = text_that_is_good_length # LLM returns this
        self.mock_model_instance.generate_content.return_value = mock_response_scenario2

        # Patch utils.clean_generated_text specifically for this scenario
        with patch.object(utils, 'clean_generated_text', return_value=text_that_is_good_length) as mock_clean_scenario2:
            with patch.object(utils, 'smart_truncate_text') as mock_smart_truncate_not_called:
                generated_text = self.generator.generate_entry(
                    target_emotion="test_no_truncation",
                    avg_word_count=target_good_length
                )
                # clean_generated_text is called with the raw output from the LLM mock
                mock_clean_scenario2.assert_called_once_with(text_that_is_good_length) 
                mock_smart_truncate_not_called.assert_not_called()
                self.assertEqual(generated_text, text_that_is_good_length)

if __name__ == '__main__':
    unittest.main() 