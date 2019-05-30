import argparse
import logging

from pepeiao.parsers import make_feature_parser, make_predict_parser, make_train_parser

LOGGER = logging.getLogger(__name__)

def make_parser():
    parser = argparse.ArgumentParser(description='A tool for using RAVEN selection tables with Keras.')
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(title='subcommands')
    make_feature_parser(subparsers.add_parser('feature', help='Preprocess audio and selection tables into feature files.'))
    make_predict_parser(subparsers.add_parser('predict', help='Predict new selection tables from an audio file.'))
    make_train_parser(subparsers.add_parser('train', help='Train a model on feature files.'))

    return parser

def _main():
    parser = make_parser()
    args = parser.parse_args()
    if args.func is None:
        args.func = lambda x: parser.print_usage()
    elif args.func == 'feature':
        import pepeiao.feature
        args.func = pepeiao.feature.main
    elif args.func == 'predict':
        import pepeiao.predict
        args.func = pepeiao.predict.main
    elif args.func == 'train':
        import pepeiao.train
        args.func = pepeiao.train.main

    args.func(args)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _main()
