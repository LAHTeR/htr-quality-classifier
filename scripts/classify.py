import argparse
import os
import sys
from pathlib import Path
from text_quality.feature.tokenizer import NautilusOcrTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Classify the quality of a (digitized) text.")

    input_args = parser.add_argument_group("Input")
    input_args.add_argument(
        "--input",
        "-i",
        type=argparse.FileType("rt"),
        nargs="*",
        default=sys.stdin,
        help="Plain text file to classify. Defaults to stdin.",
    )
    input_args.add_argument(
        "--pagexml",
        type=Path,
        nargs="*",
        help="Input file(s) in PageXML format.",
    )

    args = parser.parse_args()

    tokenizer = NautilusOcrTokenizer()

    for f in args.input:
        text = os.linesep(f.readlines())
        tokens = tokenizer.tokenize(text)
        



