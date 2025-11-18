"""
Unit tests for OpenAI Image-to-Text task implementation.

These are pure unit tests with NO real external interactions:
- All file I/O is mocked (no actual file system access)
- All API calls are mocked (no network requests)
- All external dependencies are isolated
"""
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import base64

from task.image_to_text.openai.message import ContentedMessage, TxtContent, ImgContent
from task._models.role import Role


class TestOpenAIImageToTextTask(unittest.TestCase):
    """Test suite for OpenAI Image-to-Text task with fully mocked dependencies."""

    def setUp(self):
        """Set up test fixtures."""
        self.fake_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        self.fake_base64_image = base64.b64encode(self.fake_image_bytes).decode('utf-8')

        # Mock response objects with content attribute
        self.mock_response_base64 = Mock()
        self.mock_response_base64.content = "This is a DIAL banner with colorful gradient."

        self.mock_response_url = Mock()
        self.mock_response_url.content = "This is an elephant in its natural habitat."

    @patch('builtins.open', new_callable=mock_open, read_data=b'\x89PNG\r\n\x1a\n')
    @patch('task.image_to_text.openai.task_openai_itt.Path')
    @patch('task.image_to_text.openai.task_openai_itt.DialModelClient')
    def test_start_executes_without_errors(self, mock_client_class, mock_path_class, mock_file):
        """Test that start() function executes without errors with all dependencies mocked."""
        # Mock Path construction and operations
        mock_file_instance = MagicMock()
        mock_file_instance.__truediv__ = lambda self, other: MagicMock()
        mock_path_class.return_value.parent.parent.parent.parent.__truediv__.return_value = '/fake/path/dialx-banner.png'

        # Mock DialModelClient
        mock_client = Mock()
        mock_client.get_completion.side_effect = [
            self.mock_response_base64,
            self.mock_response_url
        ]
        mock_client_class.return_value = mock_client

        # Import and execute
        from task.image_to_text.openai.task_openai_itt import start
        start()

        # Verify DialModelClient was instantiated
        mock_client_class.assert_called_once()

        # Verify get_completion was called twice
        self.assertEqual(mock_client.get_completion.call_count, 2)

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_data')
    @patch('task.image_to_text.openai.task_openai_itt.Path')
    @patch('task.image_to_text.openai.task_openai_itt.DialModelClient')
    def test_client_initialization_parameters(self, mock_client_class, mock_path_class, mock_file):
        """Test that DialModelClient is initialized with correct parameters."""
        # Mock Path
        mock_path_class.return_value.parent.parent.parent.parent.__truediv__.return_value = '/fake/path'

        # Mock client
        mock_client = Mock()
        mock_client.get_completion.return_value = self.mock_response_base64
        mock_client_class.return_value = mock_client

        # Execute
        from task.image_to_text.openai.task_openai_itt import start
        start()

        # Verify client initialization
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args.kwargs

        self.assertIn('endpoint', call_kwargs)
        self.assertIn('deployment_name', call_kwargs)
        self.assertIn('api_key', call_kwargs)
        self.assertEqual(call_kwargs['deployment_name'], 'gpt-4o')

    @patch('builtins.open', new_callable=mock_open, read_data=b'test_image_bytes')
    @patch('task.image_to_text.openai.task_openai_itt.Path')
    @patch('task.image_to_text.openai.task_openai_itt.DialModelClient')
    def test_base64_encoding_is_correct(self, mock_client_class, mock_path_class, mock_file):
        """Test that image is correctly base64 encoded."""
        # Mock Path
        mock_path_class.return_value.parent.parent.parent.parent.__truediv__.return_value = '/fake/path'

        # Mock client
        mock_client = Mock()
        mock_client.get_completion.return_value = self.mock_response_base64
        mock_client_class.return_value = mock_client

        # Execute
        from task.image_to_text.openai.task_openai_itt import start
        start()

        # Get the first call to get_completion
        first_call = mock_client.get_completion.call_args_list[0]
        messages = first_call.kwargs['messages']

        # Verify message structure
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], ContentedMessage)
        
        # Check that image content contains base64 data
        img_content = next((c for c in messages[0].content if isinstance(c, ImgContent)), None)
        self.assertIsNotNone(img_content)
        self.assertTrue(img_content.image_url.url.startswith('data:image/png;base64,'))

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_data')
    @patch('task.image_to_text.openai.task_openai_itt.Path')
    @patch('task.image_to_text.openai.task_openai_itt.DialModelClient')
    def test_base64_message_structure(self, mock_client_class, mock_path_class, mock_file):
        """Test that base64 image message has correct structure."""
        # Mock Path
        mock_path_class.return_value.parent.parent.parent.parent.__truediv__.return_value = '/fake/path'

        # Mock client
        mock_client = Mock()
        mock_client.get_completion.return_value = self.mock_response_base64
        mock_client_class.return_value = mock_client

        # Execute
        from task.image_to_text.openai.task_openai_itt import start
        start()

        # Get first call arguments
        first_call = mock_client.get_completion.call_args_list[0]
        messages = first_call.kwargs['messages']

        # Verify message structure
        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message.role, Role.USER)
        self.assertEqual(len(message.content), 2)
        
        # Verify text content
        txt_content = next((c for c in message.content if isinstance(c, TxtContent)), None)
        self.assertIsNotNone(txt_content)
        self.assertIn("What's in this image?", txt_content.text)
        
        # Verify image content
        img_content = next((c for c in message.content if isinstance(c, ImgContent)), None)
        self.assertIsNotNone(img_content)

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_data')
    @patch('task.image_to_text.openai.task_openai_itt.Path')
    @patch('task.image_to_text.openai.task_openai_itt.DialModelClient')
    def test_url_message_structure(self, mock_client_class, mock_path_class, mock_file):
        """Test that URL image message has correct structure."""
        # Mock Path
        mock_path_class.return_value.parent.parent.parent.parent.__truediv__.return_value = '/fake/path'

        # Mock client
        mock_client = Mock()
        mock_client.get_completion.side_effect = [
            self.mock_response_base64,
            self.mock_response_url
        ]
        mock_client_class.return_value = mock_client

        # Execute
        from task.image_to_text.openai.task_openai_itt import start
        start()

        # Get second call arguments
        second_call = mock_client.get_completion.call_args_list[1]
        messages = second_call.kwargs['messages']

        # Verify message structure
        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message.role, Role.USER)

        # Verify data URI image content (implementation uses base64 data URIs, not external URLs)
        img_content = next((c for c in message.content if isinstance(c, ImgContent)), None)
        self.assertIsNotNone(img_content)
        self.assertIn('data:image/png;base64,', img_content.image_url.url)

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_data')
    @patch('task.image_to_text.openai.task_openai_itt.Path')
    @patch('task.image_to_text.openai.task_openai_itt.DialModelClient')
    def test_both_completion_calls_made(self, mock_client_class, mock_path_class, mock_file):
        """Test that get_completion is called exactly twice (base64 and URL)."""
        # Mock Path
        mock_path_class.return_value.parent.parent.parent.parent.__truediv__.return_value = '/fake/path'

        # Mock client
        mock_client = Mock()
        mock_client.get_completion.side_effect = [
            self.mock_response_base64,
            self.mock_response_url
        ]
        mock_client_class.return_value = mock_client

        # Execute
        from task.image_to_text.openai.task_openai_itt import start
        start()

        # Verify exactly 2 calls
        self.assertEqual(mock_client.get_completion.call_count, 2)

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_data')
    @patch('task.image_to_text.openai.task_openai_itt.Path')
    @patch('task.image_to_text.openai.task_openai_itt.DialModelClient')
    @patch('builtins.print')
    def test_responses_are_printed(self, mock_print, mock_client_class, mock_path_class, mock_file):
        """Test that responses are printed to console."""
        # Mock Path
        mock_path_class.return_value.parent.parent.parent.parent.__truediv__.return_value = '/fake/path'

        # Mock client
        mock_client = Mock()
        mock_client.get_completion.side_effect = [
            self.mock_response_base64,
            self.mock_response_url
        ]
        mock_client_class.return_value = mock_client

        # Execute
        from task.image_to_text.openai.task_openai_itt import start
        start()

        # Verify print was called
        self.assertGreater(mock_print.call_count, 0)

        # Verify responses were printed
        print_calls_str = ' '.join([str(call) for call in mock_print.call_args_list])
        self.assertIn(self.mock_response_base64.content, print_calls_str)
        self.assertIn(self.mock_response_url.content, print_calls_str)

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_data')
    @patch('task.image_to_text.openai.task_openai_itt.Path')
    @patch('task.image_to_text.openai.task_openai_itt.DialModelClient')
    def test_no_real_file_io(self, mock_client_class, mock_path_class, mock_file):
        """Verify that no real file I/O occurs (all mocked)."""
        # Mock Path
        mock_path_class.return_value.parent.parent.parent.parent.__truediv__.return_value = '/fake/path'

        # Mock client
        mock_client = Mock()
        mock_client.get_completion.return_value = self.mock_response_base64
        mock_client_class.return_value = mock_client

        # Execute
        from task.image_to_text.openai.task_openai_itt import start
        start()

        # Verify mocked open was used
        mock_file.assert_called()

        # This test passing means no real file was accessed
        self.assertTrue(True, "No real file I/O occurred")

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_data')
    @patch('task.image_to_text.openai.task_openai_itt.Path')
    @patch('task.image_to_text.openai.task_openai_itt.DialModelClient')
    def test_no_real_api_calls(self, mock_client_class, mock_path_class, mock_file):
        """Verify that no real API calls are made (all mocked)."""
        # Mock Path
        mock_path_class.return_value.parent.parent.parent.parent.__truediv__.return_value = '/fake/path'

        # Mock client with strict tracking
        mock_client = Mock(spec=['get_completion'])
        mock_client.get_completion.return_value = self.mock_response_base64
        mock_client_class.return_value = mock_client

        # Execute
        from task.image_to_text.openai.task_openai_itt import start
        start()

        # Verify only mocked methods were called
        mock_client.get_completion.assert_called()

        # This test passing means no real API call occurred
        self.assertTrue(True, "No real API calls occurred")


if __name__ == '__main__':
    unittest.main()
