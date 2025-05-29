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
        
        target_word_count_unused = 10 # This variable seems unused in the actual logic below, main targets are forced_target_word_count and target_good_length
        
        # Configure the mock Gemini response for this test
        mock_response = MagicMock(spec=genai.types.GenerateContentResponse)
        mock_response.text = raw_llm_output
        self.mock_model_instance.generate_content.return_value = mock_response
        
        with patch.object(utils, 'clean_generated_text', return_value=raw_llm_output.strip()) as mock_clean:
            # Scenario 1: Text is too long (based on 50% tolerance) and needs truncation
            text_that_is_too_long_after_cleaning = "This is cleaned but still very very very very very very long and needs truncation for sure it really does." # count_words (split) = 20
            forced_target_word_count = 10 # Target for the generator to aim for
            # Word count (20) vs target (10): 20 is > 10 * 1.5 (15), so it's outside 50% tolerance. Truncation expected.
            
            expected_max_overshoot_for_smart_truncate = int(forced_target_word_count * 0.10) # This is for smart_truncate_text's internal check
            mock_clean.return_value = text_that_is_too_long_after_cleaning 
            
            with patch.object(utils, 'smart_truncate_text', return_value="Successfully Truncated Text.") as mock_smart_truncate:
                generated_text = self.generator.generate_entry(
                    target_emotion="test_truncation", 
                    avg_word_count=forced_target_word_count
                )
                
                mock_clean.assert_called_with(raw_llm_output)
                mock_smart_truncate.assert_called_once_with(
                    text_that_is_too_long_after_cleaning, 
                    forced_target_word_count, 
                    max_overshoot_words=expected_max_overshoot_for_smart_truncate
                )
                self.assertEqual(generated_text, "Successfully Truncated Text.")

        # Scenario 2: Text is within 50% tolerance and should not be truncated
        self.mock_model_instance.generate_content.reset_mock() 
        
        text_that_is_good_length = "This text is a pretty good length, not too long not too short just right."
        # count_words (split) for text_that_is_good_length = 13
        target_good_length = 10
        # Word count (13) vs target (10): 13 is <= 10 * 1.5 (15), so it's within 50% tolerance. No truncation expected.
        
        mock_response_scenario2 = MagicMock(spec=genai.types.GenerateContentResponse)
        mock_response_scenario2.text = text_that_is_good_length 
        self.mock_model_instance.generate_content.return_value = mock_response_scenario2

        with patch.object(utils, 'clean_generated_text', return_value=text_that_is_good_length) as mock_clean_scenario2:
            with patch.object(utils, 'smart_truncate_text') as mock_smart_truncate_not_called:
                generated_text = self.generator.generate_entry(
                    target_emotion="test_no_truncation",
                    avg_word_count=target_good_length
                )
                mock_clean_scenario2.assert_called_once_with(text_that_is_good_length)
                mock_smart_truncate_not_called.assert_not_called()
                self.assertEqual(generated_text, text_that_is_good_length)

if __name__ == '__main__':
    unittest.main() 