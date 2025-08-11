#!/usr/bin/env python3
"""
Interactive CLI for tone engine
"""
import argparse
import sys
import os
from dotenv import load_dotenv

# Ensure env
load_dotenv()

# Support running as a script or as a module
try:
    # When invoked as a package module: python -m tone_engine.cli
    from .main import ToneEngine  # type: ignore
except Exception:
    # When invoked directly: python tone_engine/cli.py
    # Add script directory to path to locate main.py
    sys.path.append(os.path.dirname(__file__))
    from main import ToneEngine  # type: ignore

print("DEBUG: ANTHROPIC_API_KEY =", os.getenv("ANTHROPIC_API_KEY"))
print("DEBUG: Current working directory =", os.getcwd())
print("DEBUG: .env file exists =", os.path.exists(".env"))


def main():
    parser = argparse.ArgumentParser(description="Rephrase a text file or direct input in your own tone using your tone profile.")
    parser.add_argument("input_file", nargs="?", help="Path to the input text file (e.g., ChatGPT script). If omitted, input will be read from stdin.")
    parser.add_argument("-o", "--output", help="Optional path to save the rephrased output")
    parser.add_argument("-c", "--client", default="default", help="Client name to use for profile separation (default: default)")
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        print("Paste or type your text below. Press CTRL+D (or CTRL+Z on Windows) when done:\n")
        input_text = sys.stdin.read()

    engine = ToneEngine(client_name=args.client)

    rephrased = engine.rephrase_in_my_tone(input_text)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(rephrased)
        print(f"Rephrased text saved to {args.output}")
    else:
        print("\n--- Rephrased Text ---\n")
        print(rephrased)

if __name__ == "__main__":
    main() 