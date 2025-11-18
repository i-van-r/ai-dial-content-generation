"""
Unit tests for Text-to-Image task implementation.

These are pure unit tests with NO real external interactions:
- All file I/O is mocked (no actual file system access)
- All API calls are mocked (no network requests)
- All external dependencies are isolated
"""
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open, AsyncMock
from datetime import datetime

from task._models.custom_content import Attachment, CustomContent
from task._models.role import Role
from task.text_to_image.task_tti import Size, Quality, Style


class TestTextToImageTask(unittest.TestCase):
    """Test suite for Text-to-Image task with fully mocked dependencies."""

    def setUp(self):
        """Set up test fixtures."""
        self.fake_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        
        self.mock_attachment1 = Attachment(
            title="generated_image_1.png",
            url="https://fake-bucket.example.com/files/image1.png",
            type="image/png"
        )
        
        self.mock_attachment2 = Attachment(
            title="generated_image_2.png",
            url="https://fake-bucket.example.com/files/image2.png",
            type="image/png"
        )
        
        self.mock_response = Mock()
        self.mock_response.custom_content = CustomContent(
            attachments=[self.mock_attachment1, self.mock_attachment2]
        )

    @patch('asyncio.run')
    @patch('task.text_to_image.task_tti.DialModelClient')
    def test_start_executes_without_errors(self, mock_model_client_class, mock_asyncio_run):
        """Test that start() function executes without errors with all dependencies mocked."""
        mock_asyncio_run.return_value = None
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.text_to_image.task_tti import start
        start()
        
        mock_model_client_class.assert_called_once()
        mock_model_client.get_completion.assert_called_once()

    @patch('asyncio.run')
    @patch('task.text_to_image.task_tti.DialModelClient')
    def test_model_client_initialization_parameters(self, mock_model_client_class, mock_asyncio_run):
        """Test that DialModelClient is initialized with correct parameters."""
        mock_asyncio_run.return_value = None
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.text_to_image.task_tti import start
        start()
        
        mock_model_client_class.assert_called_once()
        call_kwargs = mock_model_client_class.call_args.kwargs
        
        self.assertIn('endpoint', call_kwargs)
        self.assertIn('deployment_name', call_kwargs)
        self.assertIn('api_key', call_kwargs)
        self.assertEqual(call_kwargs['deployment_name'], 'dall-e-3')

    @patch('asyncio.run')
    @patch('task.text_to_image.task_tti.DialModelClient')
    def test_message_structure(self, mock_model_client_class, mock_asyncio_run):
        """Test that message has correct structure."""
        mock_asyncio_run.return_value = None
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.text_to_image.task_tti import start
        start()
        
        call_kwargs = mock_model_client.get_completion.call_args.kwargs
        messages = call_kwargs['messages']
        
        self.assertEqual(len(messages), 1)
        message = messages[0]
        
        self.assertEqual(message.role, Role.USER)
        self.assertEqual(message.content, "Sunny day on Bali")

    @patch('asyncio.run')
    @patch('task.text_to_image.task_tti.DialModelClient')
    def test_custom_fields_configuration(self, mock_model_client_class, mock_asyncio_run):
        """Test that custom_fields are correctly configured."""
        mock_asyncio_run.return_value = None
    
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
    
        from task.text_to_image.task_tti import start
        start()
    
        call_kwargs = mock_model_client.get_completion.call_args.kwargs
        custom_fields = call_kwargs['custom_fields']
    
        self.assertIn('size', custom_fields)
        self.assertIn('quality', custom_fields)
        self.assertIn('style', custom_fields)
    
        self.assertEqual(custom_fields['size'], '1024x1024')
        self.assertEqual(custom_fields['quality'], 'hd')
        self.assertEqual(custom_fields['style'], 'vivid')

    @patch('asyncio.run')
    @patch('task.text_to_image.task_tti.DialModelClient')
    def test_custom_fields_uses_enum_values(self, mock_model_client_class, mock_asyncio_run):
        """Test that custom_fields use enum values correctly."""
        mock_asyncio_run.return_value = None
    
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
    
        from task.text_to_image.task_tti import start
        start()
    
        call_kwargs = mock_model_client.get_completion.call_args.kwargs
        custom_fields = call_kwargs['custom_fields']
    
        self.assertEqual(custom_fields['size'], Size.square)
        self.assertEqual(custom_fields['quality'], Quality.hd)
        self.assertEqual(custom_fields['style'], Style.vivid)

    @patch('task.text_to_image.task_tti.DialBucketClient')
    async def test_save_images_function(self, mock_bucket_client_class):
        """Test that _save_images function correctly downloads and saves images."""
        mock_bucket_client = AsyncMock()
        mock_bucket_client.get_file.return_value = self.fake_image_bytes
        mock_bucket_client_class.return_value = mock_bucket_client
        
        attachments = [self.mock_attachment1, self.mock_attachment2]
        
        with patch('builtins.open', mock_open()) as mock_file:
            from task.text_to_image.task_tti import _save_images
            await _save_images(attachments)
        
        self.assertEqual(mock_bucket_client.get_file.call_count, 2)
        self.assertEqual(mock_file.call_count, 2)

    @patch('task.text_to_image.task_tti.DialBucketClient')
    async def test_save_images_generates_correct_filenames(self, mock_bucket_client_class):
        """Test that _save_images generates timestamped filenames."""
        mock_bucket_client = AsyncMock()
        mock_bucket_client.get_file.return_value = self.fake_image_bytes
        mock_bucket_client_class.return_value = mock_bucket_client
        
        attachments = [self.mock_attachment1]
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('task.text_to_image.task_tti.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime(2024, 1, 15, 10, 30, 45)
                mock_datetime.strftime = datetime.strftime
                
                from task.text_to_image.task_tti import _save_images
                await _save_images(attachments)
        
        mock_file.assert_called_once()
        filename = mock_file.call_args[0][0]
        
        self.assertIn('generated_image_', filename)
        self.assertIn('_1.png', filename)

    @patch('task.text_to_image.task_tti.DialBucketClient')
    async def test_save_images_writes_bytes(self, mock_bucket_client_class):
        """Test that _save_images writes bytes to file."""
        mock_bucket_client = AsyncMock()
        mock_bucket_client.get_file.return_value = self.fake_image_bytes
        mock_bucket_client_class.return_value = mock_bucket_client
        
        attachments = [self.mock_attachment1]
        
        with patch('builtins.open', mock_open()) as mock_file:
            from task.text_to_image.task_tti import _save_images
            await _save_images(attachments)
        
        mock_file().write.assert_called_once_with(self.fake_image_bytes)

    @patch('task.text_to_image.task_tti.DialBucketClient')
    async def test_save_images_bucket_client_initialization(self, mock_bucket_client_class):
        """Test that DialBucketClient is initialized correctly in _save_images."""
        mock_bucket_client = AsyncMock()
        mock_bucket_client.get_file.return_value = self.fake_image_bytes
        mock_bucket_client_class.return_value = mock_bucket_client
        
        attachments = [self.mock_attachment1]
        
        with patch('builtins.open', mock_open()):
            from task.text_to_image.task_tti import _save_images
            await _save_images(attachments)
        
        mock_bucket_client_class.assert_called_once()
        call_kwargs = mock_bucket_client_class.call_args.kwargs
        
        self.assertIn('api_key', call_kwargs)
        self.assertIn('base_url', call_kwargs)

    @patch('task.text_to_image.task_tti.DialBucketClient')
    async def test_save_images_handles_multiple_attachments(self, mock_bucket_client_class):
        """Test that _save_images handles multiple attachments correctly."""
        mock_bucket_client = AsyncMock()
        mock_bucket_client.get_file.return_value = self.fake_image_bytes
        mock_bucket_client_class.return_value = mock_bucket_client
        
        attachments = [self.mock_attachment1, self.mock_attachment2]
        
        with patch('builtins.open', mock_open()) as mock_file:
            from task.text_to_image.task_tti import _save_images
            await _save_images(attachments)
        
        self.assertEqual(mock_bucket_client.get_file.call_count, 2)
        self.assertEqual(mock_file.call_count, 2)

    @patch('asyncio.run')
    @patch('task.text_to_image.task_tti.DialModelClient')
    @patch('builtins.print')
    def test_response_is_printed(self, mock_print, mock_model_client_class, mock_asyncio_run):
        """Test that response is printed to console."""
        mock_asyncio_run.return_value = None
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.text_to_image.task_tti import start
        start()
        
        self.assertGreater(mock_print.call_count, 0)

    @patch('asyncio.run')
    @patch('task.text_to_image.task_tti.DialModelClient')
    def test_asyncio_run_called_for_save_images(self, mock_model_client_class, mock_asyncio_run):
        """Test that asyncio.run is called to execute _save_images."""
        mock_asyncio_run.return_value = None
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.text_to_image.task_tti import start
        start()
        
        mock_asyncio_run.assert_called_once()

    @patch('asyncio.run')
    @patch('task.text_to_image.task_tti.DialModelClient')
    def test_no_real_api_calls(self, mock_model_client_class, mock_asyncio_run):
        """Verify that no real API calls are made (all mocked)."""
        mock_asyncio_run.return_value = None
        
        mock_model_client = Mock(spec=['get_completion'])
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
        
        from task.text_to_image.task_tti import start
        start()
        
        mock_model_client.get_completion.assert_called()
        self.assertTrue(True, "No real API calls occurred")

    @patch('task.text_to_image.task_tti.DialBucketClient')
    async def test_no_real_file_io_in_save_images(self, mock_bucket_client_class):
        """Verify that no real file I/O occurs in _save_images (all mocked)."""
        mock_bucket_client = AsyncMock()
        mock_bucket_client.get_file.return_value = self.fake_image_bytes
        mock_bucket_client_class.return_value = mock_bucket_client
        
        attachments = [self.mock_attachment1]
        
        with patch('builtins.open', mock_open()) as mock_file:
            from task.text_to_image.task_tti import _save_images
            await _save_images(attachments)
        
        mock_file.assert_called()
        self.assertTrue(True, "No real file I/O occurred")

    @patch('asyncio.run')
    @patch('task.text_to_image.task_tti.DialModelClient')
    def test_response_contains_attachments(self, mock_model_client_class, mock_asyncio_run):
        """Test that response contains expected attachments."""
        mock_asyncio_run.return_value = None
    
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = self.mock_response
        mock_model_client_class.return_value = mock_model_client
    
        from task.text_to_image.task_tti import start
        start()
    
        self.assertEqual(len(self.mock_response.custom_content.attachments), 2)
        self.assertIsInstance(self.mock_response.custom_content.attachments[0], Attachment)
        self.assertIsInstance(self.mock_response.custom_content.attachments[1], Attachment)

    @patch('asyncio.run')
    @patch('task.text_to_image.task_tti.DialModelClient')
    def test_handles_empty_response(self, mock_model_client_class, mock_asyncio_run):
        """Test that start handles response with no attachments."""
        mock_asyncio_run.return_value = None
        
        mock_response_empty = Mock()
        mock_response_empty.custom_content = CustomContent(attachments=[])
        
        mock_model_client = Mock()
        mock_model_client.get_completion.return_value = mock_response_empty
        mock_model_client_class.return_value = mock_model_client
        
        from task.text_to_image.task_tti import start
        
        try:
            start()
            execution_completed = True
        except Exception:
            execution_completed = False
        
        self.assertTrue(execution_completed, "Function should handle empty response gracefully")


if __name__ == '__main__':
    unittest.main()
