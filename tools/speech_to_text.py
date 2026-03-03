#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech recognition module using faster-whisper
Converts WAV audio files to SRT subtitle format
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from datetime import timedelta

import yaml


@dataclass
class FasterWhisperConfig:
    """Configuration for faster-whisper speech recognition."""

    # Model settings
    model_name: str = "large-v3"
    device: str = "cpu"
    compute_type: str = "float32"

    # VAD settings
    vad_filter: bool = True
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 1000
    max_speech_duration_s: int = 30
    min_silence_duration_ms: int = 600

    # Transcription settings
    language: Optional[str] = None  # "auto" for auto-detection
    task: str = "transcribe"
    temperature: float = 0.0
    best_of: int = 5
    beam_size: int = 5

    # Output settings
    word_timestamps: bool = False
    max_words_per_line: int = 20
    clean_output: bool = True

    # Prompt settings
    initial_prompt: str = ""

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FasterWhisperConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Extract relevant fields from nested YAML structure
        model_cfg = config_dict.get("model", {})
        vad_cfg = config_dict.get("vad", {})
        trans_cfg = config_dict.get("transcription", {})
        output_cfg = config_dict.get("output", {})
        prompt_cfg = config_dict.get("prompt", {})

        return cls(
            model_name=model_cfg.get("name", "large-v3"),
            device=model_cfg.get("device", "cpu"),
            compute_type=model_cfg.get("compute_type", "float32"),
            vad_filter=vad_cfg.get("filter", True),
            vad_threshold=vad_cfg.get("threshold", 0.5),
            min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 1000),
            max_speech_duration_s=vad_cfg.get("max_speech_duration_s", 30),
            min_silence_duration_ms=vad_cfg.get("min_silence_duration_ms", 600),
            language=trans_cfg.get("language"),
            task=trans_cfg.get("task", "transcribe"),
            temperature=trans_cfg.get("temperature", 0.0),
            best_of=trans_cfg.get("best_of", 5),
            beam_size=trans_cfg.get("beam_size", 5),
            word_timestamps=output_cfg.get("word_timestamps", False),
            max_words_per_line=output_cfg.get("max_words_per_line", 20),
            clean_output=output_cfg.get("clean_output", True),
            initial_prompt=prompt_cfg.get("initial_prompt", ""),
        )


def ms_to_time_string(ms: int) -> str:
    """Convert milliseconds to SRT time format (HH:MM:SS,mmm)."""
    td = timedelta(milliseconds=ms)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def create_srt_from_segments(segments: List[Dict]) -> str:
    """
    Create SRT format string from transcription segments.

    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys

    Returns:
        SRT format string
    """

    def format_time(ms: float) -> str:
        """Convert milliseconds to SRT time format."""
        td = timedelta(milliseconds=int(ms))
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    srt_lines = []
    transcription = list(segments)
    for i, segment in enumerate(transcription, 1):
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"].strip()

        if text:
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between subtitles

    return "\n".join(srt_lines)


def transcribe_audio(
    audio_path: str, config: FasterWhisperConfig, output_path: Optional[str] = None
) -> str:
    """
    Transcribe audio file using faster-whisper.

    Args:
        audio_path: Path to input WAV file
        config: FasterWhisperConfig object
        output_path: Optional path for output SRT file

    Returns:
        SRT format string
    """
    # Import here to avoid loading if not needed
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper not installed. Install with: pip install faster-whisper"
        )

    print(f"[Speech Recognition] Loading model: {config.model_name} on {config.device}")

    # Load model
    model = WhisperModel(
        config.model_name, device=config.device, compute_type=config.compute_type
    )

    # Prepare segments iterator
    print("[Speech Recognition] Starting transcription...")

    segments, info = model.transcribe(
        audio_path,
        language=config.language
        if config.language and config.language != "auto"
        else None,
        task=config.task,
        vad_filter=config.vad_filter,
        vad_parameters=dict(
            threshold=config.vad_threshold,
            min_speech_duration_ms=config.min_speech_duration_ms,
            min_silence_duration_ms=config.min_silence_duration_ms,
        )
        if config.vad_filter
        else None,
        temperature=config.temperature,
        best_of=config.best_of,
        beam_size=config.beam_size,
        word_timestamps=config.word_timestamps,
        initial_prompt=config.initial_prompt if config.initial_prompt else None,
    )

    print(f"[Speech Recognition] Detected language: {info.language}")

    # Convert segments to list and filter empty ones
    transcription = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            transcription.append(
                {
                    "start": segment.start * 1000,  # Convert to milliseconds
                    "end": segment.end * 1000,
                    "text": text,
                }
            )

    # Create SRT output
    srt_content = create_srt_from_segments(transcription)

    # Save to file if output path provided
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        print(f"[Speech Recognition] SRT saved to: {output_path}")

    return srt_content


def get_output_path(audio_path: str) -> str:
    """
    Generate output SRT file path from audio file path.

    Args:
        audio_path: Path to WAV file (e.g., "/path/to/abc.wav")

    Returns:
        Output SRT path (e.g., "/path/to/abc.zh-cn.srt")
    """
    audio_path = Path(audio_path)
    output_path = audio_path.with_name(f"{audio_path.stem}.zh-cn.srt")
    return str(output_path)


def main():
    """Command-line interface for speech recognition."""
    parser = argparse.ArgumentParser(
        description="Speech recognition using faster-whisper to generate SRT subtitles"
    )
    parser.add_argument("audio_file", help="Path to input WAV file")
    parser.add_argument(
        "-c",
        "--config",
        default="faster_whisper_config.yaml",
        help="Path to YAML configuration file (default: faster_whisper_config.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output SRT file path (default: <input_prefix>.zh-cn.srt)",
    )
    parser.add_argument("--model", help="Model name (overrides config)")
    parser.add_argument(
        "--language", help="Source language code (overrides config, e.g., 'en', 'zh')"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], help="Device to run on (overrides config)"
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    config_path = args.config
    if not Path(config_path).is_absolute():
        # Try to find config relative to script location
        script_dir = Path(__file__).parent
        config_path = script_dir / config_path

    if Path(config_path).exists():
        print(f"[Configuration] Loading from: {config_path}")
        config = FasterWhisperConfig.from_yaml(str(config_path))
    else:
        print(f"[Configuration] Config file not found, using defaults")
        config = FasterWhisperConfig()

    # Override with command line arguments
    if args.model:
        config.model_name = args.model
    if args.language:
        config.language = args.language
    if args.device:
        config.device = args.device

    # Determine output path
    output_path = args.output
    if not output_path:
        output_path = get_output_path(args.audio_file)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Run transcription
    try:
        srt_content = transcribe_audio(
            audio_path=args.audio_file, config=config, output_path=output_path
        )
        print(f"[Success] Transcription completed. Output: {output_path}")
    except Exception as e:
        print(f"[Error] Transcription failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
