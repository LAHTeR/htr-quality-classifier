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
from joblib import Parallel
from joblib import delayed
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


class OutputRow(TypedDict):
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
        help="Output scores and text statistics.",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
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

    with Parallel(n_jobs=args.workers) as parallel:
        text_inputs: dict[str, str] = dict(
            zip(
                [file.name for file in args.input],
                parallel(
                    delayed(lambda f: os.linesep.join(f.readlines()))(file)
                    for file in args.input
                ),
            )
        )

        pagexml_inputs: dict[Path, Page] = {
            path: pagexml
            for path, pagexml in zip(
                chain(args.pagexml, glob.glob(args.pagexml_glob)),
                parallel(
                    delayed(Page.from_file)(path)
                    for path in chain(args.pagexml, glob.glob(args.pagexml_glob))
                ),
            )
            if pagexml
        }

        if args.output_scores:
            rows = (
                OutputRow(filename=filename, quality_class=quality_class)
                for filename, quality_class in zip(
                    (text_inputs | pagexml_inputs).keys(),
                    parallel(
                        delayed(pipeline.classify)(page)
                        for page in tqdm(
                            (text_inputs | pagexml_inputs).values(),
                            desc="Processing",
                            unit="file",
                        )
                    ),
                )
            )
        else:
            rows = (
                OutputRow(filename=filename, quality_class=quality_class)
                | classifier_scores
                for filename, (quality_class, classifier_scores) in tqdm(
                    zip(
                        (text_inputs | pagexml_inputs).keys(),
                        parallel(
                            delayed(pipeline.classify_with_scores)(page)
                            for page in (text_inputs | pagexml_inputs).values()
                        ),
                    ),
                    desc="Processing",
                    unit="file",
                )
            )

    fieldnames = list(OutputRow.__annotations__.keys())
    if args.output_scores:
        fieldnames += list(ClassifierScores.__annotations__.keys())

    writer = csv.DictWriter(args.output, fieldnames=fieldnames)
    writer.writeheader()

    writer.writerows(rows)
