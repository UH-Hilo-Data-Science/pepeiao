import argparse
import concurrent.futures
import csv
import itertools
import logging
import random

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
                        help = 'Desired proportion of "one" labels in training/validation data')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help = 'Training batch size')
    parser.add_argument('model', choices=pepeiao.util.get_models())
    parser.add_argument('description_file', type=argparse.FileType('r'),
                        help = 'File describing training data. A tab-delimited file with two columns: wav_filename, selection_file')
    parser.add_argument('output', help='Filename for fitted model in (.h5)')
    return parser


def data_generator(train_list, width, offset, desired_prop_ones=None):
    """Generate data by asynchronously processing wav files into spectrograms."""
    count_total = 0
    count_ones = 0
    keep_prob = 1.0
    keep = True

    if desired_prop_ones and (desired_prop_ones < 0 or desired_prop_ones > 1):
        raise ValueError('desired proportion of ones is not a valid proportion.')

    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        train_list = itertools.cycle(train_list)
        current_feature = executor.submit(pepeiao.feature.Spectrogram, *next(train_list))
        for wav_file, selection_file in train_list:
            next_future = executor.submit(pepeiao.feature.Spectrogram,
                                          wav_file, selection_file)
            current_spectrogram = current_feature.result()
            current_spectrogram.set_windowing(width, offset)
            
            if desired_prop_ones is None:
                yield from current_spectrogram.shuffled_windows()
            else:
                for window, label in current_spectrogram.shuffled_windows():
                    if label >= 0.5:
                        keep = True
                        count_ones += 1
                    else:
                        keep = random.random() < keep_prob
                    if keep:
                        count_total += 1
                        yield (window, label)

                    current_prop_ones = count_ones / count_total
                    if abs(desired_prop_ones - current_prop_ones) > 0.05:
                        if current_prop_ones - desired_prop_ones > 0.05:
                            keep_prob = min(keep_prob + 0.05, 1.0)
                        elif current_prop_ones - desired_prop_ones < -0.05:
                            keep_prob = max(0.0, keep_prob - 0.05)
                            
            current_feature = next_future

def grouper(iterable, n, fillvalue=None):
    'Collect data into fixed-length chunks or blocks (from itertools recipes)'
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def _main():
    parser = _make_parser()
    logging.basicConfig(level=logging.INFO)
                
    args = parser.parse_args()
    _LOGGER.debug(args)
    training_list = list(csv.reader(args.description_file, delimiter='\t'))

    if args.num_validation >= 1.0:
        if args.num_validation > len(training_list):
            raise ValueEror('--num-validation argument is greater than the number of available files')
        n_valid = int(args.num_validation)
    elif args.num_validation >= 0.0:
        n_valid = int(args.num_validation * len(training_list))
    else:
        raise ValueError('--num-validation argument is not positive')

    if n_valid > 0.3 * len(training_list):
        _LOGGER.warn('Using more than 30% of files as validation data.')

    training_set = grouper(
        data_generator(training_list[:-n_valid], args.width, args.offset, args.proportion_ones),
        args.batch_size)
    
    validation_set = grouper(
        data_generator(training_list[-n_valid:], args.width, args.offset, args.proportion_ones),
        args.batch_size)

    model_description = pepeiao.util.get_models()[args.model]

    input_shape = next(training_set)[0].shape[1:]
    
    model = model_description[model](input_shape)
    
    history = model.fit_generator(
        training_set,
        steps_per_epoch=200,
        shuffle=False,
        epochs=100,
        verbose=1, #0-silent, 1-progessbar, 2-1line
        validation_data=validation_set,
        validation_steps=200,
        callbacks=[EarlyStopping(patience=5)],
    )
    
    model.save(args.output)

if __name__ == '__main__':
    _main()

    
