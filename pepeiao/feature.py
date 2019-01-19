import logging
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
    def __init__(self, filename):
        super().__init__(self)
        self.read_wav(filename)

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
        return self._data[:, index:(index.self.width)]

    def _get_label(self, index, percentile=75):
        return np.percentile(self.labels[index:(index+self.width)], percentile)
        
    def read_wav(self, filename):
        """Read audio from file and compute spectrogram."""
        _LOGGER.info('Reading %s.', filename)
        samples, samp_rate = librosa.load(filename, sr=None)

        if samp_rate != _SAMP_RATE:
            _LOGGER.warn('Resampling from %s to %s Hz.', samp_rate, _SAMP_RATE)
            samples = librosa.core.resample(samples, self.sampling_rate, _SAMP_RATE)
            
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
        
    
