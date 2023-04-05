import argparse
import csv
import glob
import logging
import os
import sys
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
from text_quality.settings import CLASSIFIER_DIR
from text_quality.settings import DICTS_DIR
from text_quality.settings import HUNSPELL_DIR
from text_quality.settings import LOG_LEVEL
from text_quality.settings import QGRAMS_DIR


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
        help="Plain text file to classify. Use '-' for stdin.",
    )
    input_args.add_argument(
        "--pagexml",
        type=Path,
        nargs="*",
        default=[],
        help="Input file(s) in PageXML format.",
    )
    input_args.add_argument(
        "--pagexml-glob",
        "--glob",
        default="",
        type=str,
        help="A pattern to find a set of PageXML files, e.g. 'pagexml/*.xml'.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("wt"),
        default=sys.stdout,
        help="Output file; defaults to stdout.",
    )
    parser.add_argument(
        "--output-scores",
        action="store_true",
        help="Output scores and text statistics.",
    )
    args = parser.parse_args()

    ### INITIALIZE
    # TODO: move to settings.py
    hunspell_language = "nl"
    token_dict_file: Path = DICTS_DIR / "nl_voc.txt"
    qgrams_file: Path = QGRAMS_DIR / "nl_voc.txt"
    pipeline_file: Path = CLASSIFIER_DIR / "pipeline_nn.joblib"

    for file in (token_dict_file, qgrams_file, pipeline_file):
        assert file.is_file()

    tokenizer = NautilusOcrTokenizer()

    featurizer = Featurizer(
        Scorers(
            dict_score=HunspellDictionary.from_path(HUNSPELL_DIR, hunspell_language),
            dict_score_gt=TokenDictionary.from_file(token_dict_file),
            n_gram_score=QGram.from_file(qgrams_file),
            garbage_score=GarbageDetector(),
        ),
        tokenizer=tokenizer,
    )
    pipeline = Pipeline.from_file(pipeline_file, featurizer)

    text_inputs = {f.name: os.linesep.join(f.readlines()) for f in args.input}

    pagexml_inputs = {
        pagexml.name: Page.from_file(pagexml).get_text() for pagexml in args.pagexml
    }
    pagexml_glob_inputs = {
        file: Page.from_file(Path(file)).get_text()
        for file in glob.glob(args.pagexml_glob)
    }

    if args.output_scores:
        fieldnames = (
            OutputRow.__annotations__ | ClassifierScores.__annotations__
        ).keys()
    else:
        fieldnames = OutputRow.__annotations__.keys()

    writer = csv.DictWriter(args.output, fieldnames=fieldnames)
    writer.writeheader()

    for name, text in tqdm(
        (text_inputs | pagexml_inputs | pagexml_glob_inputs).items(),
        desc="Processing",
        unit="file",
    ):
        if args.output_scores:
            quality_class, classifier_scores = pipeline.classifiy_with_scores(text)
            row = (
                OutputRow(filename=name, quality_class=quality_class)
                | classifier_scores
            )
        else:
            row = OutputRow(filename=name, quality_class=pipeline.classify(text))

        writer.writerow(row)
