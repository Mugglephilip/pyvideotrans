# AGENTS.md — Development Guidelines for pyVideoTrans

## Project Overview
pyVideoTrans is a video translation/dubbing tool with speech recognition, translation, and TTS pipelines.

## Build & Environment

### Package Manager
- **Use `uv`** for dependency management (NOT pip/conda)
- Python: **3.10 only** (specified in pyproject.toml)
- Install: `uv sync` | Add package: `uv add <package>`

### Running Code
```bash
# GUI
uv run sp.py

# CLI
uv run cli.py --task vtv --name "./video.mp4" --source_language_code zh --target_language_code en

# Tests
uv run testcuda.py  # Only test file in repo
```

### CUDA Setup (Optional)
```bash
uv remove torch torchaudio
uv add torch==2.7 torchaudio==2.7 --index-url https://download.pytorch.org/whl/cu128
```

## Code Style & Conventions

### Imports (3-tier organization)
```python
# 1. Standard library
import asyncio
import json
from pathlib import Path
from typing import List, Dict

# 2. Third-party
import httpx
from pydub import AudioSegment
from tenacity import retry, stop_after_attempt

# 3. Local (videotrans.*)
from videotrans.configure import config
from videotrans.configure.config import logger, settings
from videotrans.util import tools
```

### Naming Conventions
- **Classes**: PascalCase (`BaseRecogn`, `EdgeTTS`, `VideoTransError`)
- **Functions**: snake_case (`get_speech_timestamp`, `_signal`)
- **Variables**: snake_case (`audio_file`, `cache_folder`)
- **Constants**: UPPER_SNAKE_CASE (`ROOT_DIR`, `TEMP_DIR`, `NO_RETRY_EXCEPT`)
- **Private methods**: Leading underscore (`_vad_split`, `_create_audio_with_retry`)

### Type Hints
- **Required** for function parameters and return types
- Use `Optional[str]` for nullable types
- Use `List[Type]`, `Dict[K, V]` from `typing` module
- `Union[X, Y]` for multiple types
- `Any` only when absolutely necessary

### Dataclasses (Preferred Pattern)
```python
@dataclass
class EdgeTTS(BaseTTS):
    def __post_init__(self):
        super().__post_init__()
        # Initialization logic
        
    async def _create_audio(self, item, index):
        # Implementation
```

### Error Handling

#### Exception Hierarchy
```python
class VideoTransError(Exception):
    """Base exception"""
    
class SpeechToTextError(VideoTransError): pass
class TranslateSrtError(VideoTransError): pass
class DubbSrtError(VideoTransError): pass
```

#### Retry Pattern (tenacity)
```python
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_not_exception_type
from videotrans.configure._except import NO_RETRY_EXCEPT, StopRetry

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_not_exception_type(NO_RETRY_EXCEPT),
    before=before_log(logger, logging.DEBUG),
    after=after_log(logger, logging.DEBUG)
)
def api_call():
    pass
```

#### Exception Handling
```python
try:
    result = await api_call()
except NO_RETRY_EXCEPT as e:
    # Don't retry - re-raise or handle immediately
    raise StopRetry(f"Permanent error: {e.message}")
except VideoTransError as e:
    # Custom app errors
    logger.error(f"Error: {e.message}")
    raise
except Exception as e:
    # Generic fallback
    logger.exception(f"Unexpected error: {e}")
    raise VideoTransError(str(e))
```

### Logging
```python
from videotrans.configure.config import logger

# Use logger (pre-configured in config.py)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.exception("Exception with traceback")  # Use in except blocks
```

## Architecture Patterns

### Base Classes
- **Recognition**: `videotrans.recognition._base.BaseRecogn`
- **Translation**: `videotrans.translator._base.BaseTrans`
- **TTS**: `videotrans.tts._base.BaseTTS`

Extends these base classes for new implementations.

### Component Structure
```
videotrans/
├── configure/      # Config, settings, exceptions
├── recognition/    # Speech-to-text implementations
├── translator/     # Translation implementations
├── tts/            # Text-to-speech implementations
├── task/           # Task orchestration
├── util/           # Utilities (tools, ffmpeg, http)
└── component/      # UI components (PySide6)
```

### Async Patterns
```python
# Semaphore for concurrency control
semaphore = asyncio.Semaphore(10)

async def process_with_semaphore(item):
    async with semaphore:
        return await process(item)

# Event for stop signal
self._stop_event = asyncio.Event()
if self._stop_event.is_set():
    return  # Exit early
```

## Internationalization (i18n)
```python
from videotrans.configure.config import tr
message = tr("gui_qyd缨正译")  # Key lookup from locale files
```

## File Paths
- Use `pathlib.Path` (not `os.path`)
- Root: `ROOT_DIR = Path(__file__).parent.parent.parent`
- Temp: `TEMP_DIR = f'{ROOT_DIR}/tmp/_temp'`
- Check existence: `tools.vail_file(path)`

## Database/State
Use `settings` dict for global configuration:
```python
from videotrans.configure.config import settings

threshold = float(settings.get('threshold', 0.5))
max_concurrent = int(settings.get('edgetts_max_concurrent_tasks', 10))
```

## Testing
- Limited test coverage in current codebase
- Manual testing via GUI/CLI
- For new features: add inline validation, not separate test files

## Git & Workflows
- CI: GitHub Actions (`.github/workflows/main.yml`)
- PyInstaller builds for Windows distribution
- No pre-commit hooks configured

## Common Pitfalls
1. **Don't use `as any`** - Type safety is enforced
2. **Don't suppress exceptions** - Let errors propagate with context
3. **Use tenacity for retries** - Don't implement manual retry loops
4. **Check `_stop_event`** - Async tasks must respect cancellation
5. **Match import order** - Standard lib → Third-party → Local
6. **No blank imports** - Group imports with blank lines between sections
