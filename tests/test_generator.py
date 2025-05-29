import unittest
from unittest.mock import patch, MagicMock
import os
import sys

SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

try:
    from generator import JournalGenerator
    # We will also need utils for the generator's internal calls, but we mock the LLM part.
    import utils 
except ImportError:
    print("Failed to import from generator or utils. Ensure PYTHONPATH is set correctly.")
    if 'generator' not in sys.modules or 'utils' not in sys.modules:
        raise

class TestJournalGenerator(unittest.TestCase):

    def setUp(self):
        """Set up for each test. We mock the LLM pipeline to avoid actual model loading."""
        # Mock the transformers.pipeline call within the JournalGenerator's __init__
        self.mock_pipeline = MagicMock()
        # Simulate a successful pipeline creation that returns a callable object
        # The callable object (mock_pipeline itself) should return a list with a dict containing 'generated_text'
        self.mock_pipeline.return_value = [{'generated_text': 'This is a mock LLM response.'}]
        
        # Patch 'pipeline' in the 'transformers' library specifically where JournalGenerator looks for it.
        # This needs to be the string path to the function as it's imported/used in generator.py
        # Corrected: Patch where 'pipeline' is looked up by the 'generator' module.
        # Assuming 'from transformers import pipeline' is used in generator.py
        self.pipeline_patcher = patch('generator.pipeline', return_value=self.mock_pipeline) 
        self.mock_transformers_pipeline = self.pipeline_patcher.start()
        
        # Now, when JournalGenerator is instantiated, it will use our mock pipeline
        try:
            self.generator = JournalGenerator(model_name="mock_model") # model_name doesn't matter due to patching
        except Exception as e:
            self.fail(f"JournalGenerator instantiation failed even with mocked pipeline: {e}")

    def tearDown(self):
        """Clean up after each test."""
        self.pipeline_patcher.stop() # Important to stop the patch

    def test_generator_initialization(self):
        """Test that the JournalGenerator initializes (with a mocked pipeline)."""
        self.assertIsNotNone(self.generator)
        self.assertIsNotNone(self.generator.llm_pipeline, "LLM pipeline should be set on generator.")
        # Check that transformers.pipeline was called by JournalGenerator's __init__
        # The actual call is made to the patched object, which is 'generator.pipeline'
        self.mock_transformers_pipeline.assert_called_once_with("text-generation", model="mock_model", device=-1)

    def test_construct_prompt_messages_basic(self):
        """Test the basic prompt construction without examples."""
        target_emotion = "curious"
        avg_word_count = 50
        messages = self.generator._construct_prompt_messages(target_emotion, avg_word_count)
        
        self.assertEqual(len(messages), 2) # System and User message
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        self.assertIn(target_emotion, messages[1]['content'])
        self.assertIn(str(avg_word_count), messages[1]['content'])
        self.assertIn("New Journal Entry:", messages[1]['content'])
        self.assertNotIn("Example 1:", messages[1]['content'])

    def test_construct_prompt_messages_with_examples(self):
        """Test prompt construction with few-shot examples."""
        target_emotion = "excited"
        avg_word_count = 70
        example_entries = ["Example entry 1 about excitement.", "Example entry 2, also very exciting!"]
        messages = self.generator._construct_prompt_messages(target_emotion, avg_word_count, example_entries)
        
        self.assertEqual(messages[1]['role'], 'user')
        user_content = messages[1]['content']
        self.assertIn(target_emotion, user_content)
        self.assertIn(str(avg_word_count), user_content)
        self.assertIn("Example 1:", user_content)
        self.assertIn(example_entries[0], user_content)
        self.assertIn("Example 2:", user_content)
        self.assertIn(example_entries[1], user_content)
        self.assertIn("New Journal Entry:", user_content)

    def test_generate_entry_calls_pipeline(self):
        """Test that generate_entry calls the LLM pipeline with correct parameters."""
        target_emotion = "nostalgic"
        avg_word_count = 60
        max_tokens = 90
        
        # Redefine what the mock pipeline call returns for this specific test if needed
        # The one in setUp is a default. This would override it if generate_entry was called multiple times
        # with different expected outputs.
        self.generator.llm_pipeline.return_value = [{'generated_text': 'A nostalgic piece from mock LLM.'}]

        generated_text = self.generator.generate_entry(target_emotion, avg_word_count, max_new_tokens=max_tokens)
        
        self.generator.llm_pipeline.assert_called_once() # Check it was called
        # Inspect the arguments it was called with
        args_call, kwargs_call = self.generator.llm_pipeline.call_args
        
        # args_call[0] should be the `messages` list
        self.assertIsInstance(args_call[0], list)
        self.assertEqual(args_call[0][1]['role'], 'user') # User message
        self.assertIn(target_emotion, args_call[0][1]['content'])
        
        # kwargs_call should contain max_new_tokens and num_return_sequences
        self.assertEqual(kwargs_call.get('max_new_tokens'), max_tokens)
        self.assertEqual(kwargs_call.get('num_return_sequences'), 1)
        
        self.assertEqual(generated_text, "A nostalgic piece from mock LLM.") # Assuming simple cleaning

    def test_generate_entry_handles_llm_failure(self):
        """Test that generate_entry returns empty string if LLM call fails."""
        self.generator.llm_pipeline.side_effect = Exception("LLM simulated error")
        generated_text = self.generator.generate_entry("any_emotion", 50)
        self.assertEqual(generated_text, "")

    def test_generate_entry_handles_unexpected_llm_response_format(self):
        """Test handling of unexpected response format from LLM."""
        self.generator.llm_pipeline.return_value = [{"wrong_key": "some text"}] # Bad format
        generated_text = self.generator.generate_entry("any_emotion", 50)
        self.assertEqual(generated_text, "")

        self.generator.llm_pipeline.return_value = "just a string, not a list" # Another bad format
        generated_text = self.generator.generate_entry("any_emotion", 50)
        self.assertEqual(generated_text, "")

    def test_generated_text_cleaning_and_truncation(self):
        """Test that generated text is cleaned and truncated."""
        raw_llm_output = "  New Journal Entry: This is a mock response that is deliberately a bit too long for the target word count. It needs to be truncated.  "
        expected_word_count = 10
        # We rely on the mock_pipeline to return this, then utils to clean/truncate
        self.generator.llm_pipeline.return_value = [{'generated_text': raw_llm_output}]
        
        # Mock utils.truncate_to_word_count to check it gets called with expected input
        # and to control its output for this test.
        with patch.object(utils, 'truncate_to_word_count', return_value="Truncated text.") as mock_truncate:
            with patch.object(utils, 'clean_generated_text', return_value="Cleaned text, but still long.") as mock_clean:
                
                generated_text = self.generator.generate_entry(
                    target_emotion="test", 
                    avg_word_count=expected_word_count
                )
                
                mock_clean.assert_called_once_with(raw_llm_output.strip()[len("New Journal Entry:"):].strip()) # Check input to clean
                # The input to truncate is the output of clean
                mock_truncate.assert_called_once_with("Cleaned text, but still long.", expected_word_count, tolerance=int(expected_word_count * 0.2))
                self.assertEqual(generated_text, "Truncated text.")

if __name__ == '__main__':
    unittest.main() 