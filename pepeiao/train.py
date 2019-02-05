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
    parser.add_argument('model', choices=pepeiao.util.get_models())
    parser.add_argument('description_file', type=argparse.FileType('r'),
                        help = 'File describing training data. A tab-delimited file with two columns: wav_filename, selection_file')
    return parser


def data_generator(train_list, width, offset, desired_prop_ones=None):
    """Generate data by asynchronously processing wav files into spectrograms."""
    count_total = 0
    count_ones = 0
    keep_prob = 1.0
    keep = True
    
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
                        keep = random.random() < keep_prob:
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
            
            

if __name__ == '__main__':
    parser = _make_parser()
    logging.basicConfig(level=logging.INFO)
                
    args = parser.parse_args()
    print(args)

    training_list = list(csv.reader(args.description_file, delimiter='\t'))
    # for item in training_list:
    #     print(item)
                          
    try:
        for idx, item in enumerate(data_generator(training_list, args.width, args.offset)):
            if idx % 100 == 0:
                print('.', end='', flush=True)
    except KeyboardInterrupt:
        pass

