import argparse
import logging

import pepeiao.parsers

LOGGER = logging.getLogger(__name__)

def make_parser():
    parser = argparse.ArgumentParser(description='A tool for using RAVEN selection tables with Keras.')
    parser.set_defaults(func=parser.print_usage)
    subparsers = parser.add_subparsers(title='subcommands')
    pepeiao.parsers.feature_parser(subparsers.add_parser('feature', help='Preprocess audio and selection tables into feature files.'))
    pepeiao.parsers.predict_parser(subparsers.add_parser('predict', help='Predict new selection tables from an audio file.'))
    pepeiao.parsers.train_parser(subparsers.add_parser('train', help='Train a model on feature files.'))

    return parser

def _main():
    parser = make_parser()
    args = parser.parse_args()
    args.func()

if __name__ == '__main__':
    logging.basicConfig()
    _main()
