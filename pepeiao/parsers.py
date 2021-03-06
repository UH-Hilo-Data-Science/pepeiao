import argparse

import pepeiao.util

def make_feature_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    # parser.add_argument('-o', '--output', default=None,
    #                     help='Directory to write features to')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('wav', nargs='+')
    parser.set_defaults(func='feature')
    return parser

def make_predict_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('model', help="fitted model file")
    parser.add_argument('-s', '--selections', default=None,
                        help="look for selections table of same name and write roc data")
    parser.add_argument('wav', nargs='+', help="wav files to predict on")
    parser.set_defaults(func='predict')
    return parser


def make_train_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--width', type=float, default=0.5)
    parser.add_argument('-o', '--offset', type=float, default=0.125)
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-n', '--num-validation', type=float, default=0.25,
                        help='An integer number of training files to use as a validation set, or if less than one, use as a proportion.')
    parser.add_argument('-p', '--proportion-ones', type=float,
                        help='Desired proportion of "one" labels in training/validation data')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='Training batch size')
    parser.add_argument('model', choices=pepeiao.util.available_models())
    parser.add_argument('feature', nargs='+',
                        help='Preprocessed feature files for training')
    parser.add_argument('output', help='Filename for fitted model in (.h5)')
    parser.set_defaults(func='train')
    return parser
