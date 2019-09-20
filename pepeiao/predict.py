import argparse
import csv
import sys
import logging
import librosa
import os
import re

from pepeiao.constants import _RAVEN_HEADER, _SELECTION_KEY, _BEGIN_KEY, _END_KEY, _FILE_KEY
import pepeiao.feature
import pepeiao.models
from pepeiao.parsers import make_predict_parser as _make_parser

_LOGGER = logging.getLogger(__name__)

def predict(feature, model, out_stream=sys.stdout):
    feature.predict(model)
    writer = csv.DictWriter(out_stream, fieldnames=[
        _SELECTION_KEY, _BEGIN_KEY, _END_KEY, _FILE_KEY],
    delimiter='\t')
    writer.writeheader()
    for idx, (start, end) in enumerate(feature.time_intervals, start=1):
        writer.writerow({_SELECTION_KEY: idx, _BEGIN_KEY: '{:.3f}'.format(start), _END_KEY: '{:.3f}'.format(end), _FILE_KEY: feature.file_name})
    
def main(args):
    import keras.models
    try:
        model = keras.models.load_model(args.model, custom_objects={'_prob_bird': pepeiao.models._prob_bird})
    except OSError as err:
        print("Failed to open model file: {}".format(args.model))
        return -1
    for filename in args.wav:
        feature = pepeiao.feature.Spectrogram(filename, args.selections)
        predict(feature, model)
        if args.selections is not None:
            _LOGGER.info('Writing roc table')
            with open(os.path.basename(filename) + '.roc', 'w') as rocfile:
                print('true, pred', file=rocfile)
                for true, pred in feature.roc():
                    print(true, pred, sep=', ', file=rocfile)
                

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = _make_parser().parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print('Exiting on user interrupt.')
