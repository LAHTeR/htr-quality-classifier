#!/usr/bin/env python3

import argparse
import csv
import glob
import logging
import os
import sys
from itertools import chain
from pathlib import Path
from typing import TypedDict
from tqdm import tqdm
from text_quality.classifier.pipeline import ClassifierScores
from text_quality.classifier.pipeline import Pipeline
from text_quality.feature.featurize import Featurizer
from text_quality.feature.featurize import Scorers
from text_quality.feature.scorer.dictionary import HunspellDictionary
from text_quality.feature.scorer.dictionary import TokenDictionary
from text_quality.feature.scorer.garbage import GarbageDetector
from text_quality.feature.scorer.q_gram import QGram
from text_quality.feature.tokenizer import NautilusOcrTokenizer
from text_quality.page.page import Page
from text_quality.settings import HUNSPELL_DIR
from text_quality.settings import HUNSPELL_LANGUAGE
from text_quality.settings import LOG_LEVEL
from text_quality.settings import PIPELINE_FILE
from text_quality.settings import QGRAMS_FILE
from text_quality.settings import TOKEN_DICT_FILE


logging.basicConfig(level=LOG_LEVEL)

REASON_FIELDNAME = "Reason"


class OutputRow(TypedDict):
    """Container class for the rows in the CSV output."""

    filename: str
    quality_class: int


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Classify the quality of a (digitized) text.")

    input_args = parser.add_argument_group("Input")
    input_args.add_argument(
        "--input",
        "-i",
        type=argparse.FileType("rt"),
        nargs="*",
        default=[],
        metavar="FILE",
        help="Plain text file(s) to classify. Use '-' for stdin.",
    )
    input_args.add_argument(
        "--pagexml",
        type=Path,
        nargs="*",
        default=[],
        metavar="FILE",
        help="Input file(s) in PageXML format.",
    )
    input_args.add_argument(
        "--pagexml-glob",
        "--glob",
        default="",
        type=str,
        metavar="PATTERN",
        help="A pattern to find a set of PageXML files, e.g. 'pagexml/*.xml'.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("wt"),
        default=sys.stdout,
        metavar="FILE",
        help="Output file; defaults to stdout.",
    )
    parser.add_argument(
        "--output-scores",
        action="store_true",
        help="Output scores and text statistics, and reason for classification.",
    )
    args = parser.parse_args()

    tokenizer = NautilusOcrTokenizer()

    featurizer = Featurizer(
        Scorers(
            dict_score=HunspellDictionary.from_path(HUNSPELL_DIR, HUNSPELL_LANGUAGE),
            dict_score_gt=TokenDictionary.from_file(TOKEN_DICT_FILE),
            n_gram_score=QGram.from_file(QGRAMS_FILE),
            garbage_score=GarbageDetector(),
        ),
        tokenizer=tokenizer,
    )
    pipeline = Pipeline.from_file(PIPELINE_FILE, featurizer)
    if pipeline.features != featurizer.features:
        raise RuntimeError(
            f"Pipline input features ({pipeline.features})"
            f"do not match scorers ({featurizer.features})."
        )

    text_inputs = {f.name: os.linesep.join(f.readlines()) for f in args.input}

    pagexml_inputs = {}
    for pagexml in chain(args.pagexml, glob.glob(args.pagexml_glob)):
        if pagexml in pagexml_inputs:
            logging.warning("Duplicate input file: '%s'", pagexml)
        try:
            pagexml_inputs[pagexml] = Page.from_file(pagexml)
        except Exception as e:
            logging.error("Error parsing file '%s': %s", pagexml, str(e))
            pagexml_inputs[pagexml] = ""

    fieldnames = list(OutputRow.__annotations__.keys())
    if args.output_scores:
        fieldnames += list(ClassifierScores.__annotations__.keys()) + [REASON_FIELDNAME]

    writer = csv.DictWriter(args.output, fieldnames=fieldnames)
    writer.writeheader()

    for name, page in tqdm(
        (text_inputs | pagexml_inputs).items(), desc="Processing", unit="file"
    ):
        if args.output_scores:
            quality_class, classifier_scores, reason = pipeline.classify_with_scores(
                page
            )
            row = (
                OutputRow(filename=name, quality_class=quality_class)
                | classifier_scores
                | {REASON_FIELDNAME: reason.name}
            )
        else:
            row = OutputRow(filename=name, quality_class=pipeline.classify(page))

        writer.writerow(row)
