# Unit Tests

## Setup

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_task_openai_itt.py
```

### Run with coverage
```bash
pytest --cov=task --cov-report=html
```

### Run specific test method
```bash
pytest tests/test_task_openai_itt.py::TestOpenAIImageToTextTask::test_start_with_valid_inputs
```

## PyCharm Integration

### Running Tests in PyCharm

1. **Right-click on test file** → Select "Run 'pytest in test_task_openai_itt.py'"
2. **Run single test**: Right-click on a test method → Select "Run 'pytest for test_...'"
3. **Keyboard shortcut**: 
   - Run tests: `⌃⌥R` (Control + Option + R)
   - Debug tests: `⌃⌥D` (Control + Option + D)

### Viewing Test Results

- Green checkmark ✓: Test passed
- Red X ✗: Test failed
- Yellow triangle ⚠: Test skipped

### Test Coverage in PyCharm

1. **Run with Coverage**: Right-click test file → "More Run/Debug" → "Run with Coverage"
2. **View Coverage Report**: Coverage panel will show percentage per file
3. **Highlight Coverage**: Covered lines appear in green, uncovered in red

## Test Structure

- `test_task_openai_itt.py`: Unit tests for OpenAI Image-to-Text task
  - Tests API client instantiation
  - Tests message structure for base64 and URL formats
  - Tests environment variable handling
  - Tests output formatting
  - Tests model configuration
