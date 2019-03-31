import argparse
import concurrent.futures
import csv
import itertools
import logging
import random

import numpy as np
import keras.callbacks

import pepeiao.util

_LOGGER = logging.getLogger(__name__)

def _make_parser():
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
    parser.add_argument('model', choices=pepeiao.util.get_models())
    parser.add_argument('feature', nargs='+',
                        help='Preprocessed feature files for training')
    parser.add_argument('output', help='Filename for fitted model in (.h5)')
    return parser


def data_generator(feature_list, width, offset, batch_size=100, desired_prop_ones=None):
    count_total = 0
    count_ones = 0
    keep_prob = 1.0
    result_idx = 0
    keep = True
    windows = None
    labels = None
    if desired_prop_ones and (desired_prop_ones < 0 or desired_prop_ones > 1):
        raise ValueError('desired proportion of ones is not a valid proportion.')

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        train_list = itertools.cycle(feature_list)
        current_future = executor.submit(pepeiao.feature.load_feature, next(train_list))
        for feat_file in train_list:
            next_future = executor.submit(pepeiao.feature.load_feature, feat_file)
            current_feature = current_future.result()
            current_feature.set_windowing(width, offset)
            _LOGGER.info('Loaded feature %s.', current_feature.file_name)
            current_future = next_future

            if windows is None:  # initilize result arrays on first iteration
                shape = current_feature._get_window(0).shape
                windows = np.empty((batch_size, *shape), dtype=float)
                labels = np.empty(batch_size, dtype=float)

            ## take items from the feature and put them into the arrays until full then yield arrays
            if desired_prop_ones is None:
                for wind, lab in current_feature.shuffled_windows():
                    windows[result_idx] = wind
                    labels[result_idx] = lab
                    result_idx += 1
                    if (result_idx % batch_size) == 0:
                        result_idx = 0
                        yield windows, labels
            else: # keep track of proportion of ones
                for wind, lab in current_feature.shuffled_windows():
                    if lab >= 0.5:
                        keep = True
                        count_ones += 1
                    else:
                        keep = random.random() < keep_prob
                    if keep:
                        count_total += 1
                        windows[result_idx] = wind
                        labels[result_idx] = lab
                        result_idx += 1

                        if (result_idx % batch_size) == 0:
                            result_idx = 0
                            yield windows, labels

                    current_prop_ones = count_ones / count_total
                    if (current_prop_ones - desired_prop_ones) > 0.05:
                        keep_prob = min(keep_prob + 0.05, 1.0)
                    elif (current_prop_ones - desired_prop_ones) < 0.05:
                        keep_prob = max(keep_prob - 0.05, 0.0)


# def grouper(iterable, n, fillvalue=None):
#     'Collect data into fixed-length chunks or blocks (from itertools recipes)'
#     # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
#     args = [iter(iterable)] * n
#     return itertools.zip_longest(*args, fillvalue=fillvalue)

def _main():
    parser = _make_parser()
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    _LOGGER.debug(args)

    if args.num_validation >= 1.0:
        if args.num_validation > len(args.feature):
            raise ValueError('--num-validation argument is greater than the number of available files')
        n_valid = int(args.num_validation)
    elif args.num_validation >= 0.0:
        n_valid = int(args.num_validation * len(args.feature))
    else:
        raise ValueError('--num-validation argument is not positive')

    if n_valid > 0.3 * len(args.feature):
        _LOGGER.warning('Using more than 30% of files as validation data.')

    training_set = data_generator(args.feature[:-n_valid], args.width, args.offset,
                                  args.batch_size, args.proportion_ones)

    validation_set = data_generator(args.feature[-n_valid:], args.width, args.offset,
                                    args.batch_size, args.proportion_ones)

    model_description = pepeiao.util.get_models()[args.model]

    input_shape = next(training_set)[0].shape[1:]

    model = model_description['model'](input_shape)

    history = model.fit_generator(
        training_set,
        steps_per_epoch=200,
        shuffle=False,
        epochs=100,
        verbose=1, #0-silent, 1-progessbar, 2-1line
        validation_data=validation_set,
        validation_steps=200,
        callbacks=[keras.callbacks.EarlyStopping(patience=5)],
    )

    model.save(args.output)

if __name__ == '__main__':
    _main()
