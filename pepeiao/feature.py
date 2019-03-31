import argparse
import logging
import pickle
import random

import librosa
import numpy as np

from pepeiao.constants import _SAMP_RATE, _LABEL_THRESHOLD
from pepeiao.denoise import wp_denoise
import pepeiao.util as util

_LOGGER = logging.getLogger(__name__)


class Feature():
    def data_windows(self):
        raise NotImplementedError

    def label_windows(self):
        raise NotImplementedError

    def windows(self):
        return zip(self.data_windows(), self.label_windows())

    def shuffled_windows(self):
        raise NotImplementedError

    def infinite_shuffle(self):
        """Generate an infinite sequence of shuffled windows. Desk is exhausted before reshuffling."""
        while True:
            for x in self.shuffled_windows():
                yield x


class Spectrogram(Feature):
    def __init__(self, filename, selection_file=None):
        super().__init__()
        self.width = None
        self.stride = None
        self.labels = None
        self.read_wav(filename)
        if selection_file:
            selections = util.load_selections(selection_file)
            self.selections_to_labels(selections)
        _LOGGER.info('Spectrogram finished initialization.')


    def data_windows(self):
        """Generate the sequence of data windows."""
        for idx in range(0, self._data.shape[1], self.stride):
            yield self._get_window(idx)

    def label_windows(self):
        """Generate the sequence of labels."""
        for idx in range(0, self._data.shape[1], self.stride):
            yield self._get_label(idx)

    def shuffled_windows(self):
        """Generate the windowed data/label pairs, in random order."""
        indices = list(range(0, self._data.shape[1], self.stride))
        random.shuffle(indices)
        for idx in indices:
            yield (self._get_window(idx), self._get_label(idx))

    def _get_window(self, index):
        return self._data[:, index:(index+self.width)]

    def _get_label(self, index, percentile=75):
        return np.percentile(self.labels[index:(index+self.width)], percentile)

    def read_wav(self, filename):
        """Read audio from file and compute spectrogram."""
        _LOGGER.info('Reading %s.', filename)
        samples, samp_rate = librosa.load(filename, sr=None)

        if samp_rate != _SAMP_RATE:
            _LOGGER.warning('Resampling from %s to %s Hz.', samp_rate, _SAMP_RATE)
            samples = librosa.core.resample(samples, samp_rate, _SAMP_RATE)

        samples = wp_denoise(samples, 'dmey', level=6, scale=99.9)

        _LOGGER.info('Computing STFT')

        self.file_name = filename
        self.samp_rate = _SAMP_RATE
        self._data = librosa.stft(samples)
        self.times = librosa.frames_to_time(range(self._data.shape[1]), sr=self.samp_rate)


    def set_windowing(self, width, stride):
        """Setup windowing process (argument values in seconds)."""
        self.width = librosa.time_to_frames(width, self.samp_rate)
        self.stride = librosa.time_to_frames(stride, self.samp_rate)


    def selections_to_labels(self, selections):
        """Set the labels from a list of selections."""
        self.intervals = util.selections_to_intervals(selections)

    @property
    def intervals(self):
        """Retreive the intervals from the current label vector."""
        intervals = []
        if self.labels is not None:
            idx = 0
            start_index = None
            end_index = None
            while idx < self.labels.shape[0]:
                if self.labels[idx] >= _LABEL_THRESHOLD:
                    start_index = idx
                    try:
                        old_end = intervals[-1][1]
                        if idx < (old_end + self.stride):
                            start_index, _ = intervals.pop()
                    except IndexError:
                        pass
                    end_index = idx + 1
                    while (end_index < self.labels.shape[0]) and (self.labels[end_index] >= _LABEL_THRESHOLD):
                        end_index += 1
                    intervals.append((start_index, end_index))
                    idx = end_index
                else:
                    idx += 1
        _LOGGER.info('Constructed %d intervals from labels.', len(intervals))
        return intervals

    @intervals.setter
    def intervals(self, value):
        """Set the labels from a list of time intervals."""
        self.labels = np.array([any(util.in_interval(t, s) for s in value) for t in self.times], dtype=np.float)


def load_feature(filename):
    try:
        with open(filename, 'rb') as featfile:
            result = pickle.load(featfile)
    except IOError:
        _LOGGER.error('Failed to read feature file.')
        raise
    if not isinstance(result, Feature):
        raise ValueError('Loaded object is not a Feature')
    return result

def _make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'))
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('wav')
    parser.add_argument('selections', nargs='?')
    return parser

def main(args):
    try:
        feature = Spectrogram(args.wav)
    except FileNotFoundError:
        _LOGGER.error('Could not read %s', args.wav)
        return 1

    if args.selections:
        try:
            selections = util.load_selections(args.selections)
        except FileNotFoundError:
            _LOGGER.error('Could not read %s', args.selections)
            return 1

        feature.selections_to_labels(selections)
        _LOGGER.info('Feature data nbytes: %f', feature._data.nbytes)

    if args.output:
        try:
            pickle.dump(feature, args.output)
            print('Wrote feature to', args.output.name)
        except IOError as e:
            _LOGGER.error(e)
    return 0

if __name__ == '__main__':
    import sys
    parser = _make_parser()
    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level)
    sys.exit(main(args))
