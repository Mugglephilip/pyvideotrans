#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translation module using Ollama LLM
Translates SRT subtitle files to target language
"""

import argparse
import httpx
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

import yaml


@dataclass
class OllamaConfig:
    """Configuration for Ollama translation."""

    # API settings
    base_url: str = "http://localhost:11434"
    endpoint: str = "/api/generate"
    timeout: int = 120

    # Model settings
    model_name: str = "llama3"
    context_length: int = 8192
    max_tokens: int = 2048

    # Generation parameters
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_beams: int = 1

    # Translation settings
    target_language: str = "Chinese"
    source_language: str = ""
    system_prompt: str = """You are a professional subtitle translation engine.
Translate the following SRT subtitle content accurately while preserving the timing and structure.
Output ONLY the translated content, no explanations or additional text.
Maintain the original SRT format exactly:
- Keep line numbers unchanged
- Keep timestamps unchanged
- Translate only the subtitle text
- Preserve line breaks and formatting"""

    # Format settings
    input_format: str = "srt"
    output_format: str = "srt"
    preserve_line_numbers: bool = True
    preserve_timestamps: bool = True

    # Performance settings
    max_concurrent: int = 5
    batch_size: int = 1
    request_delay: float = 0.5

    # Retry settings
    max_retries: int = 3
    retry_delay: int = 5
    backoff_multiplier: float = 2.0

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OllamaConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Extract from nested YAML structure
        api_cfg = config_dict.get("api", {})
        model_cfg = config_dict.get("model", {})
        gen_cfg = config_dict.get("generation", {})
        trans_cfg = config_dict.get("translation", {})
        fmt_cfg = config_dict.get("format", {})
        perf_cfg = config_dict.get("performance", {})
        retry_cfg = config_dict.get("retry", {})

        return cls(
            base_url=api_cfg.get("base_url", "http://localhost:11434"),
            endpoint=api_cfg.get("endpoint", "/api/generate"),
            timeout=api_cfg.get("timeout", 120),
            model_name=model_cfg.get("name", "llama3"),
            context_length=model_cfg.get("context_length", 8192),
            max_tokens=model_cfg.get("max_tokens", 2048),
            temperature=gen_cfg.get("temperature", 0.3),
            top_p=gen_cfg.get("top_p", 0.9),
            top_k=gen_cfg.get("top_k", 40),
            repeat_penalty=gen_cfg.get("repeat_penalty", 1.1),
            num_beams=gen_cfg.get("num_beams", 1),
            target_language=trans_cfg.get("target_language", "Chinese"),
            source_language=trans_cfg.get("source_language", ""),
            system_prompt=trans_cfg.get(
                "system_prompt", cls.__dataclass_fields__["system_prompt"].default
            ),
            input_format=fmt_cfg.get("input_format", "srt"),
            output_format=fmt_cfg.get("output_format", "srt"),
            preserve_line_numbers=fmt_cfg.get("preserve_line_numbers", True),
            preserve_timestamps=fmt_cfg.get("preserve_timestamps", True),
            max_concurrent=perf_cfg.get("max_concurrent", 5),
            batch_size=perf_cfg.get("batch_size", 1),
            request_delay=perf_cfg.get("request_delay", 0.5),
            max_retries=retry_cfg.get("max_retries", 3),
            retry_delay=retry_cfg.get("retry_delay", 5),
            backoff_multiplier=retry_cfg.get("backoff_multiplier", 2.0),
        )


def parse_srt(srt_content: str) -> List[Dict]:
    """
    Parse SRT content into list of subtitle dictionaries.

    Args:
        srt_content: SRT format string

    Returns:
        List of dicts with keys: line, time, text
    """
    subtitles = []
    blocks = srt_content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            line_num = lines[0].strip()
            time_line = lines[1].strip()
            text = "\n".join(lines[2:]).strip()

            subtitles.append({"line": line_num, "time": time_line, "text": text})

    return subtitles


def merge_srt(subtitles: List[Dict]) -> str:
    """
    Merge subtitle dictionaries back to SRT format.

    Args:
        subtitles: List of subtitle dicts

    Returns:
        SRT format string
    """
    blocks = []
    for sub in subtitles:
        block = f"{sub['line']}\n{sub['time']}\n{sub['text']}"
        blocks.append(block)

    return "\n\n".join(blocks)


def translate_srt_with_ollama(
    srt_content: str, config: OllamaConfig, source_lang: Optional[str] = None
) -> str:
    """
    Translate SRT content using Ollama LLM.

    Args:
        srt_content: SRT content to translate
        config: OllamaConfig object
        source_lang: Optional source language code

    Returns:
        Translated SRT content
    """
    # Determine source language
    if not source_lang:
        source_lang = config.source_language
    if not source_lang:
        # Try to detect from content or default to auto
        source_lang = "detected automatically"

    # Build the prompt
    if source_lang.strip():
        prompt = f"""{config.system_prompt}

Source Language: {source_lang}
Target Language: {config.target_language}

Please translate the following SRT subtitles from {source_lang} to {config.target_language}:

<srt_content>
{srt_content}
</srt_content>

Output ONLY the translated SRT content:"""
    else:
        prompt = f"""{config.system_prompt}

Target Language: {config.target_language}

Please translate the following SRT subtitles to {config.target_language}:

<srt_content>
{srt_content}
</srt_content>

Output ONLY the translated SRT content:"""

    # Make API request
    url = f"{config.base_url}{config.endpoint}"

    payload = {
        "model": config.model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repeat_penalty": config.repeat_penalty,
            "num_predict": config.max_tokens,
        },
    }

    print(f"[Translation] Sending request to Ollama at {url}")
    print(f"[Translation] Model: {config.model_name}, Target: {config.target_language}")

    # Make API request without proxy
    try:
        with httpx.Client(timeout=config.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as e:
        print(f"[Error] HTTP error: {e}")
        raise
    except httpx.RequestError as e:
        print(f"[Error] Request failed: {e}")
        raise

    # Extract translated content
    translated = result.get("response", "")

    # Try to extract SRT content from response
    # Remove any markdown code blocks if present
    translated = re.sub(r"```(?:srt)?\s*", "", translated)
    translated = re.sub(r"```", "", translated)

    # Parse as SRT to validate
    try:
        parsed = parse_srt(translated.strip())
        if parsed:
            print(f"[Translation] Successfully translated {len(parsed)} subtitle lines")
            return merge_srt(parsed)
        else:
            print(
                f"[Warning] Translation result doesn't look like valid SRT, returning raw response"
            )
            return translated.strip()
    except Exception as e:
        print(f"[Warning] Error parsing SRT: {e}")
        return translated.strip()


def translate_file(
    input_path: str,
    output_path: str,
    config: OllamaConfig,
    source_lang: Optional[str] = None,
):
    """
    Translate SRT file using Ollama.

    Args:
        input_path: Path to input SRT file
        output_path: Path to output SRT file
        config: OllamaConfig object
        source_lang: Optional source language code
    """
    # Read input SRT
    with open(input_path, "r", encoding="utf-8") as f:
        srt_content = f.read()

    print(f"[Translation] Input file: {input_path}")
    block_count = len(srt_content.split("\n\n"))
    print(f"[Translation] Subtitle blocks: {block_count}")

    # Translate
    translated = translate_srt_with_ollama(
        srt_content=srt_content, config=config, source_lang=source_lang
    )

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)

    print(f"[Translation] Output saved to: {output_path}")


def main():
    """Command-line interface for SRT translation."""
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles using Ollama LLM"
    )
    parser.add_argument("input_file", help="Path to input SRT file")
    parser.add_argument(
        "-c",
        "--config",
        default="ollama_config.yaml",
        help="Path to YAML configuration file (default: ollama_config.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output SRT file path (default: <input_prefix>.translated.srt)",
    )
    parser.add_argument("--model", help="Model name (overrides config)")
    parser.add_argument("--target-lang", help="Target language (overrides config)")
    parser.add_argument(
        "--source-lang", help="Source language (optional, for better translation)"
    )
    parser.add_argument("--url", help="Ollama API base URL (overrides config)")

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    config_path = args.config
    if not Path(config_path).is_absolute():
        script_dir = Path(__file__).parent
        config_path = script_dir / config_path

    if Path(config_path).exists():
        print(f"[Configuration] Loading from: {config_path}")
        config = OllamaConfig.from_yaml(str(config_path))
    else:
        print(f"[Configuration] Config file not found, using defaults")
        config = OllamaConfig()

    # Override with command line arguments
    if args.model:
        config.model_name = args.model
    if args.target_lang:
        config.target_language = args.target_lang
    if args.url:
        config.base_url = args.url
    if args.source_lang:
        config.source_language = args.source_lang

    # Determine output path
    output_path = args.output
    if not output_path:
        input_path = Path(args.input_file)
        output_path = input_path.with_name(f"{input_path.stem}.translated.srt")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Translate with retries
    for attempt in range(1, config.max_retries + 1):
        try:
            print(f"\n[Translation] Attempt {attempt}/{config.max_retries}")
            translate_file(
                input_path=args.input_file,
                output_path=str(output_path),
                config=config,
                source_lang=args.source_lang,
            )
            print(f"[Success] Translation completed!")
            sys.exit(0)
        except Exception as e:
            print(f"[Error] Translation attempt {attempt} failed: {e}", file=sys.stderr)
            if attempt < config.max_retries:
                import time

                delay = config.retry_delay * (
                    config.backoff_multiplier ** (attempt - 1)
                )
                print(f"[Retry] Waiting {delay}s before next attempt...")
                time.sleep(delay)
            else:
                print(
                    f"[Error] All {config.max_retries} attempts failed", file=sys.stderr
                )
                sys.exit(1)


if __name__ == "__main__":
    main()
