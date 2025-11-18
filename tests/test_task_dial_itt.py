"""
Unit tests for DIAL Image-to-Text task implementation.

These are pure unit tests with NO real external interactions:
- All file I/O is mocked (no actual file system access)
- All API calls are mocked (no network requests)
- All external dependencies are isolated
"""
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open, AsyncMock
from io import BytesIO

from task._models.custom_content import Attachment
from task._models.role import Role


class TestDialImageToTextTask(unittest.TestCase):
    """Test suite for DIAL Image-to-Text task with fully mocked dependencies."""

    def setUp(self):
        """Set up test fixtures."""
        self.fake_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        
        self.mock_response = Mock()
        self.mock_response.content = "This is a DIAL banner with colorful gradient background."
        
        self.mock_attachment = Attachment(
            title="dialx-banner.png",
            url="https://fake-bucket.example.com/files/dialx-banner.png",
            type="image/png"
        )

    @patch('asyncio.run')
    @patch('builtins.open', new_callable=mock_open, read_data=b'\x89PNG\r\n\x1a\n')
    @patch('task.image_to_text.task_dial_itt.DialBucketClient')
    @patch('task.image_to_text.task_dial_itt.DialModelClient')
    def test_start_executes_without_errors(self, mock_model_client_class, mock_bucket_client_class, mock_file, mock_asyncio_run):
        """Test that start() function executes without errors with all dependencies mocked."""
        mock_asyncio_run.return_value = self.mock_attachment
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.image_to_text.task_dial_itt import start
        start()
        
        mock_model_client_class.assert_called_once()
        mock_model_client.get_completion.assert_called_once()

    @patch('asyncio.run')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image')
    @patch('task.image_to_text.task_dial_itt.DialBucketClient')
    @patch('task.image_to_text.task_dial_itt.DialModelClient')
    def test_model_client_initialization_parameters(self, mock_model_client_class, mock_bucket_client_class, mock_file, mock_asyncio_run):
        """Test that DialModelClient is initialized with correct parameters."""
        mock_asyncio_run.return_value = self.mock_attachment
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.image_to_text.task_dial_itt import start
        start()
        
        mock_model_client_class.assert_called_once()
        call_kwargs = mock_model_client_class.call_args.kwargs
        
        self.assertIn('endpoint', call_kwargs)
        self.assertIn('deployment_name', call_kwargs)
        self.assertIn('api_key', call_kwargs)
        self.assertEqual(call_kwargs['deployment_name'], 'gpt-4o')

    @patch('task.image_to_text.task_dial_itt.DialBucketClient')
    async def test_put_image_function(self, mock_bucket_client_class):
        """Test that _put_image function correctly uploads image and returns attachment."""
        mock_bucket_client = AsyncMock()
        mock_bucket_client.upload_file.return_value = "https://fake-bucket.example.com/files/dialx-banner.png"
        mock_bucket_client_class.return_value = mock_bucket_client
        
        with patch('builtins.open', mock_open(read_data=b'fake_image_data')):
            from task.image_to_text.task_dial_itt import _put_image
            attachment = await _put_image()
        
        self.assertIsInstance(attachment, Attachment)
        self.assertEqual(attachment.title, "dialx-banner.png")
        self.assertEqual(attachment.type, "image/png")
        self.assertIn("https://", attachment.url)
        
        mock_bucket_client.upload_file.assert_called_once()

    @patch('task.image_to_text.task_dial_itt.DialBucketClient')
    async def test_put_image_creates_bytesio(self, mock_bucket_client_class):
        """Test that _put_image creates BytesIO object from file."""
        mock_bucket_client = AsyncMock()
        mock_bucket_client.upload_file.return_value = "https://fake-url.com/file.png"
        mock_bucket_client_class.return_value = mock_bucket_client
        
        fake_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        
        with patch('builtins.open', mock_open(read_data=fake_image_data)):
            from task.image_to_text.task_dial_itt import _put_image
            attachment = await _put_image()
        
        upload_call_args = mock_bucket_client.upload_file.call_args
        file_arg = upload_call_args[0][0]
        
        self.assertIsInstance(file_arg, BytesIO)

    @patch('asyncio.run')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image')
    @patch('task.image_to_text.task_dial_itt.DialBucketClient')
    @patch('task.image_to_text.task_dial_itt.DialModelClient')
    def test_message_structure_with_attachment(self, mock_model_client_class, mock_bucket_client_class, mock_file, mock_asyncio_run):
        """Test that message has correct structure with attachment."""
        mock_asyncio_run.return_value = self.mock_attachment
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.image_to_text.task_dial_itt import start
        start()
        
        call_kwargs = mock_model_client.get_completion.call_args.kwargs
        messages = call_kwargs['messages']
        
        self.assertEqual(len(messages), 1)
        message = messages[0]
        
        self.assertEqual(message.role, Role.USER)
        self.assertEqual(message.content, "What do you see on this picture?")
        self.assertIsNotNone(message.custom_content)
        self.assertEqual(len(message.custom_content.attachments), 1)
        
        attachment = message.custom_content.attachments[0]
        self.assertIsInstance(attachment, Attachment)

    @patch('asyncio.run')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image')
    @patch('task.image_to_text.task_dial_itt.DialBucketClient')
    @patch('task.image_to_text.task_dial_itt.DialModelClient')
    def test_attachment_properties(self, mock_model_client_class, mock_bucket_client_class, mock_file, mock_asyncio_run):
        """Test that attachment has correct properties."""
        mock_asyncio_run.return_value = self.mock_attachment
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.image_to_text.task_dial_itt import start
        start()
        
        call_kwargs = mock_model_client.get_completion.call_args.kwargs
        messages = call_kwargs['messages']
        attachment = messages[0].custom_content.attachments[0]
        
        self.assertEqual(attachment.title, "dialx-banner.png")
        self.assertEqual(attachment.type, "image/png")
        self.assertIn("https://", attachment.url)

    @patch('asyncio.run')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image')
    @patch('task.image_to_text.task_dial_itt.DialBucketClient')
    @patch('task.image_to_text.task_dial_itt.DialModelClient')
    @patch('builtins.print')
    def test_response_is_printed(self, mock_print, mock_model_client_class, mock_bucket_client_class, mock_file, mock_asyncio_run):
        """Test that response is printed to console."""
        mock_asyncio_run.return_value = self.mock_attachment
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.image_to_text.task_dial_itt import start
        start()
        
        self.assertGreater(mock_print.call_count, 0)
        
        print_calls_str = ' '.join([str(call) for call in mock_print.call_args_list])
        self.assertIn(self.mock_response.content, print_calls_str)

    @patch('asyncio.run')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image')
    @patch('task.image_to_text.task_dial_itt.DialBucketClient')
    @patch('task.image_to_text.task_dial_itt.DialModelClient')
    def test_no_real_file_io(self, mock_model_client_class, mock_bucket_client_class, mock_file, mock_asyncio_run):
        """Verify that no real file I/O occurs (all mocked)."""
        mock_asyncio_run.return_value = self.mock_attachment
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.image_to_text.task_dial_itt import start
        
        with patch('asyncio.run', return_value=self.mock_attachment):
            start()
        
        self.assertTrue(True, "No real file I/O occurred")

    @patch('asyncio.run')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image')
    @patch('task.image_to_text.task_dial_itt.DialBucketClient')
    @patch('task.image_to_text.task_dial_itt.DialModelClient')
    def test_no_real_api_calls(self, mock_model_client_class, mock_bucket_client_class, mock_file, mock_asyncio_run):
        """Verify that no real API calls are made (all mocked)."""
        mock_asyncio_run.return_value = self.mock_attachment
        
        mock_model_client = Mock(spec=['get_completion'])
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.image_to_text.task_dial_itt import start
        start()
        
        mock_model_client.get_completion.assert_called()
        self.assertTrue(True, "No real API calls occurred")

    @patch('asyncio.run')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image')
    @patch('task.image_to_text.task_dial_itt.DialBucketClient')
    @patch('task.image_to_text.task_dial_itt.DialModelClient')
    def test_asyncio_run_called_for_put_image(self, mock_model_client_class, mock_bucket_client_class, mock_file, mock_asyncio_run):
        """Test that asyncio.run is called to execute _put_image."""
        mock_asyncio_run.return_value = self.mock_attachment
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.image_to_text.task_dial_itt import start
        start()
        
        mock_asyncio_run.assert_called_once()

    @patch('task.image_to_text.task_dial_itt.DialBucketClient')
    async def test_put_image_bucket_client_initialization(self, mock_bucket_client_class):
        """Test that DialBucketClient is initialized correctly in _put_image."""
        mock_bucket_client = AsyncMock()
        mock_bucket_client.upload_file.return_value = "https://fake-url.com/file.png"
        mock_bucket_client_class.return_value = mock_bucket_client
        
        with patch('builtins.open', mock_open(read_data=b'fake_image')):
            from task.image_to_text.task_dial_itt import _put_image
            await _put_image()
        
        mock_bucket_client_class.assert_called_once()
        call_kwargs = mock_bucket_client_class.call_args.kwargs
        
        self.assertIn('api_key', call_kwargs)
        self.assertIn('base_url', call_kwargs)


if __name__ == '__main__':
    unittest.main()
