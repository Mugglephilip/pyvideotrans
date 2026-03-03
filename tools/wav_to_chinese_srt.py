#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click WAV to Chinese Subtitle Translation Tool

This tool provides a complete workflow:
1. Speech recognition using faster-whisper
2. Subtitle translation to Chinese using Ollama LLM
3. Output as SRT format

Usage:
    python wav_to_chinese_srt.py input.wav

Or with custom configs:
    python wav_to_chinese_srt.py input.wav --fw-config faster_whisper_config.yaml --ollama-config ollama_config.yaml
"""

import argparse
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import our modules
from speech_to_text import (
    FasterWhisperConfig,
    transcribe_audio,
    get_output_path as get_srt_path,
)
from translate_srt import OllamaConfig, translate_file


def run_pipeline(
    wav_path: str,
    fw_config: FasterWhisperConfig,
    ollama_config: OllamaConfig,
    output_srt_path: Optional[str] = None,
    skip_translation: bool = False,
    skip_speech_recognition: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the complete pipeline: WAV -> Speech Recognition -> Translation -> Chinese SRT

    Args:
        wav_path: Path to input WAV file
        fw_config: Faster-whisper configuration
        ollama_config: Ollama configuration
        output_srt_path: Optional output path for final SRT
        skip_translation: If True, stop after speech recognition
        skip_speech_recognition: If True, expect input SRT file instead of WAV
        verbose: Print progress messages

    Returns:
        dict with keys: raw_srt_path, translated_srt_path, status
    """
    wav_path = Path(wav_path)
    result = {"raw_srt_path": None, "translated_srt_path": None, "status": "pending"}

    def log(msg, level="info"):
        if verbose:
            prefix = {
                "info": "[INFO]",
                "success": "[SUCCESS]",
                "error": "[ERROR]",
                "warning": "[WARNING]",
            }.get(level, "[INFO]")
            print(f"{prefix} {msg}")

    try:
        # Step 1: Speech Recognition (WAV -> SRT)
        if not skip_speech_recognition:
            log(f"Starting speech recognition for: {wav_path.name}")
            log(f"Using model: {fw_config.model_name} on {fw_config.device}")

            # Generate output path for raw SRT
            raw_srt_path = get_srt_path(str(wav_path))
            # Change extension to .zh-cn.srt as requested
            raw_srt_path = str(wav_path.parent / f"{wav_path.stem}.zh-cn.srt")

            # Run speech recognition
            srt_content = transcribe_audio(
                audio_path=str(wav_path), config=fw_config, output_path=raw_srt_path
            )

            if not srt_content.strip():
                result["status"] = "failed"
                result["error"] = "Speech recognition produced empty output"
                log("Speech recognition failed: empty output", "error")
                return result

            result["raw_srt_path"] = raw_srt_path
            log(f"Raw SRT saved to: {raw_srt_path}", "success")
        else:
            # Input is already an SRT file
            raw_srt_path = str(wav_path)
            result["raw_srt_path"] = raw_srt_path
            log(f"Using existing SRT: {raw_srt_path}")

        # Step 2: Translation (if not skipped)
        if not skip_translation:
            log("Starting translation to Chinese...")

            # Determine output path for translated SRT
            if output_srt_path:
                translated_path = output_srt_path
            else:
                # Use the same .zh-cn.srt extension
                translated_path = raw_srt_path

            # Translate using Ollama
            translate_file(
                input_path=raw_srt_path,
                output_path=translated_path,
                config=ollama_config,
                source_lang=fw_config.language
                if fw_config.language and fw_config.language != "auto"
                else None,
            )

            result["translated_srt_path"] = translated_path
            log(f"Translated SRT saved to: {translated_path}", "success")
        else:
            # No translation, raw SRT is the final output
            result["translated_srt_path"] = result["raw_srt_path"]

        result["status"] = "success"
        log("Pipeline completed successfully!", "success")
        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        log(f"Pipeline failed: {e}", "error")
        import traceback

        if verbose:
            traceback.print_exc()
        return result


def main():
    """Command-line interface for one-click WAV to Chinese SRT conversion."""
    parser = argparse.ArgumentParser(
        description="One-click tool to convert WAV audio to Chinese SRT subtitles. "
        "Uses faster-whisper for speech recognition and Ollama LLM for translation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - auto-detect language, translate to Chinese
  python wav_to_srt.py interview.wav
  
  # Specify source language for better recognition
  python wav_to_srt.py interview.wav --language en
  
  # Use custom config files
  python wav_to_srt.py audio.wav --fw-config my_whisper.yaml --ollama-config my_ollama.yaml
  
  # Only run speech recognition (no translation)
  python wav_to_srt.py audio.wav --skip-translation
  
  # Translate an existing SRT file
  python wav_to_srt.py input.srt --skip-speech-recognition
        """,
    )

    parser.add_argument(
        "input_file",
        help="Path to input WAV file (or SRT file if using --skip-speech-recognition)",
    )

    # Faster-whisper config
    parser.add_argument(
        "--fw-config",
        default="faster_whisper_config.yaml",
        help="Path to faster-whisper YAML config (default: faster_whisper_config.yaml)",
    )

    # Ollama config
    parser.add_argument(
        "--ollama-config",
        default="ollama_config.yaml",
        help="Path to Ollama YAML config (default: ollama_config.yaml)",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        help="Output SRT file path (default: <input_prefix>.zh-cn.srt)",
    )

    # Override options
    parser.add_argument(
        "--language",
        help="Source language code (e.g., 'en', 'zh', 'ja'). Auto-detect if not specified",
    )
    parser.add_argument("--model", help="Whisper model name (overrides config)")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device for Whisper (overrides config)",
    )
    parser.add_argument("--ollama-url", help="Ollama API URL (overrides config)")
    parser.add_argument("--ollama-model", help="Ollama model name (overrides config)")
    parser.add_argument(
        "--target-lang",
        default="Chinese",
        help="Target language for translation (default: Chinese)",
    )

    # Skip options
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip translation, only run speech recognition",
    )
    parser.add_argument(
        "--skip-speech-recognition",
        action="store_true",
        help="Skip speech recognition (input is already SRT)",
    )

    # Misc
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet mode (minimal output)"
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Check file extension
    if not args.skip_speech_recognition:
        if input_path.suffix.lower() not in [
            ".wav",
            ".mp3",
            ".m4a",
            ".flac",
            ".aac",
            ".ogg",
        ]:
            print(
                f"Warning: Input file is {input_path.suffix}, expected audio format like .wav",
                file=sys.stderr,
            )
    else:
        if input_path.suffix.lower() != ".srt":
            print(
                f"Error: For --skip-speech-recognition, input must be .srt file",
                file=sys.stderr,
            )
            sys.exit(1)

    verbose = args.verbose and not args.quiet

    # Load configs
    fw_config_path = Path(args.fw_config)
    if not fw_config_path.is_absolute():
        fw_config_path = Path(__file__).parent / args.fw_config

    ollama_config_path = Path(args.ollama_config)
    if not ollama_config_path.is_absolute():
        ollama_config_path = Path(__file__).parent / args.ollama_config

    if fw_config_path.exists():
        if verbose:
            print(f"[Config] Loading faster-whisper config: {fw_config_path}")
        fw_config = FasterWhisperConfig.from_yaml(str(fw_config_path))
    else:
        if verbose:
            print(f"[Config] Faster-whisper config not found, using defaults")
        fw_config = FasterWhisperConfig()

    if ollama_config_path.exists():
        if verbose:
            print(f"[Config] Loading Ollama config: {ollama_config_path}")
        ollama_config = OllamaConfig.from_yaml(str(ollama_config_path))
    else:
        if verbose:
            print(f"[Config] Ollama config not found, using defaults")
        ollama_config = OllamaConfig()

    # Apply command-line overrides
    if args.language:
        fw_config.language = args.language
    if args.model:
        fw_config.model_name = args.model
    if args.device:
        fw_config.device = args.device
    if args.ollama_url:
        ollama_config.base_url = args.ollama_url
    if args.ollama_model:
        ollama_config.model_name = args.ollama_model
    if args.target_lang:
        ollama_config.target_language = args.target_lang

    # Print pipeline info
    if verbose:
        print("\n" + "=" * 60)
        print("WAV to Chinese SRT Pipeline")
        print("=" * 60)
        print(f"Input: {input_path}")
        print(f"Output: {args.output or '<input_prefix>.zh-cn.srt'}")
        if args.skip_speech_recognition:
            print("Mode: Translation only (input is SRT)")
        elif args.skip_translation:
            print("Mode: Speech recognition only (no translation)")
        else:
            print("Mode: Full pipeline (Speech Recognition + Translation)")
        print(f"Source Language: {fw_config.language or 'auto-detect'}")
        print(f"Target Language: {ollama_config.target_language}")
        print("=" * 60 + "\n")

    # Run pipeline
    start_time = datetime.now()

    result = run_pipeline(
        wav_path=str(input_path),
        fw_config=fw_config,
        ollama_config=ollama_config,
        output_srt_path=args.output,
        skip_translation=args.skip_translation,
        skip_speech_recognition=args.skip_speech_recognition,
        verbose=verbose,
    )

    # Report results
    elapsed = (datetime.now() - start_time).total_seconds()

    if verbose:
        print("\n" + "=" * 60)
        if result["status"] == "success":
            print(f"Status: SUCCESS")
            print(f"Time elapsed: {elapsed:.1f}s")
            if result.get("raw_srt_path"):
                print(f"Raw SRT: {result['raw_srt_path']}")
            if result.get("translated_srt_path"):
                print(f"Final SRT: {result['translated_srt_path']}")
        else:
            print(f"Status: FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
        print("=" * 60)

    # Exit with appropriate code
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
