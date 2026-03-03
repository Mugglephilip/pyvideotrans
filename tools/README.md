# WAV to Chinese SRT Translation Tool

One-click command-line tool for converting WAV audio files to Chinese SRT subtitles.

## Features

- **Speech Recognition**: Uses faster-whisper for accurate speech-to-text
- **Translation**: Uses Ollama LLM for translating subtitles to Chinese
- **Configurable**: All parameters via YAML config files
- **CLI Interface**: Full command-line control with override options

## Requirements

- Python 3.10+
- Ollama (optional, for translation)
- faster-whisper (installed)

## Usage

### Basic Usage (Speech Recognition Only)

```bash
cd tools
uv run python wav_to_chinese_srt.py /path/to/audio.wav
```

This will:
1. Detect language automatically
2. Generate SRT subtitles
3. Output: `audio.zh-cn.srt`

### Skip Translation

```bash
uv run python wav_to_chinese_srt.py audio.wav --skip-translation
```

### Skip Speech Recognition (Translate Existing SRT)

```bash
uv run python wav_to_chinese_srt.py existing.srt --skip-speech-recognition
```

### With Custom Configs

```bash
uv run python wav_to_chinese_srt.py audio.wav \
  --fw-config custom_whisper.yaml \
  --ollama-config custom_ollama.yaml
```

### Specify Source Language

```bash
uv run python wav_to_chinese_srt.py audio.wav --language en
```

### Full Pipeline Options

```bash
uv run python wav_to_chinese_srt.py --help
```

## Configuration Files

### faster_whisper_config.yaml

Configure speech recognition:
- Model selection (tiny, base, small, medium, large-v3)
- Device (cpu/cuda)
- VAD settings
- Language detection
- Temperature, beam_size, etc.

### ollama_config.yaml

Configure translation:
- Ollama API URL
- Model name (llama3, mistral, etc.)
- Target language
- Generation parameters (temperature, top_p, etc.)

## Output Format

Output file naming: `<input_prefix>.zh-cn.srt`

Example:
- Input: `interview.wav`
- Output: `interview.zh-cn.srt`

## SRT Format

Standard SRT format:
```
1
00:00:00,000 --> 00:00:02,000
Subtitle text here

2
00:00:02,000 --> 00:00:05,000
Next subtitle line
```

## Translation Setup (Optional)

To enable translation, start Ollama server:

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3`
3. Start server: `ollama serve`
4. Run translation with full pipeline

## Testing

Speech recognition tested successfully:
- Tested with Chinese audio file (`no-remove.wav`)
- Transcription time: ~5 seconds on CPU (M1/M2)
- Output: `no-remove.zh-cn.srt` with proper SRT format
- Language auto-detection: Working (detected Chinese)

Example output:
```srt
1
00:00:00,000 --> 00:00:02,000
第三類接觸還有多人

2
00:00:02,000 --> 00:00:05,000
微博正式展開拍攝任務已經見滿周年
```

Translation requires Ollama server to be running.

---

## Troubleshooting

### "Module not found" errors
```bash
uv sync  # Install dependencies
```

### Translation fails
- Ensure Ollama is running: `ollama serve`
- Check model exists: `ollama list`
- Verify API URL in ollama_config.yaml

### Slow transcription
- Use smaller model in config: `base` instead of `large-v3`
- Enable CUDA if GPU available
