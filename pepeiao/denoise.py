import argparse
import itertools
import logging

import librosa
from matplotlib.cm import get_cmap
import numpy as np
from PIL import Image
import pywt

_LOGGER = logging.getLogger(__name__)


def best_level():
    pass

def denoise(samples, wavelet, level, scale):
    """Denoise based on wavelets. See wp_denoise for more info."""

    decomp = pywt.wavedec(samples, wavelet, mode='reflect', level=level)
    threshold = np.percentile(np.abs(decomp[1]), 90)
    _LOGGER.info('Threshold: %f', threshold)
    decomp[1:] = (pywt.threshold(i, value=threshold, mode='garotte') for i in decomp[1:]) # don't threshold approximation coefs
    # decomp[:] = (pywt.threshold(i, value=threshold, mode='garotte') for i in decomp[:]) # threshold approximation coefs
    return pywt.waverec(decomp, wavelet, mode='reflect')

def wp_denoise(samples, wavelet, level, scale):
    """Denoise a signal using packet wavelets.

    Adapted from Priyadarshani, N., et al.,
    'Birdsong Denoising Using Wavelets' PLOS One (2016) 
    https://doi.org/10.1371/journal.pone.0146790
"""
    
    wpt = pywt.WaveletPacket(data=samples, wavelet=wavelet, maxlevel=level)
    threshold = np.percentile(np.abs(wpt['d'].data), scale)
    wpt.walk(_threshold, kwargs=dict(value=threshold, mode='garotte'))
    wpt.reconstruct(update=True)
    return wpt.data

def _threshold(node, *args, **kwargs):
    """Threshold node data and return True, suitable for use with tree walk."""
    node.data = pywt.threshold(node.data, *args, **kwargs)
    return True


def write_spectrogram(samples, filename, colormap='viridis', gamma=1.0):
    _LOGGER.info('Computing stft spectrogram.')
    spectrogram = librosa.stft(samples)
    arr = np.flipud(np.abs(spectrogram[:, :2000]))
    arr -= arr.min()
    arr *= (1/np.percentile(arr,99))
    arr **= gamma
    arr = get_cmap(colormap)(arr)
    arr *= 255
    try:
        Image.fromarray(arr.astype('uint8')).save(filename)
        _LOGGER.info('Wrote %s', filename)
    except IOError:
        _LOGGER.warn('Failed to write spectrogram image.')

    
def shannon(samples):
    x = samples ** 2
    value = -((x * np.nan_to_num(np.log(x))).sum())
    _LOGGER.info('Shannon entropy: %s', value)
    return 
    
    
def _make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    return parser

def _main(args):
    samples, samp_rate = librosa.load(args.file, sr=None)
    _LOGGER.info('Read samples from %s, sampling rate: %s Hz', args.file, samp_rate)
    _LOGGER.info('Samples object shape: %s', samples.shape)
    
    denoised_samples = wp_denoise(samples, wavelet='dmey', level=6, scale=99.9)
    
    write_spectrogram(samples, 'orig.png')
    write_spectrogram(denoised_samples, 'denoised.png')
    librosa.output.write_wav('denoised.wav', denoised_samples, sr=samp_rate)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = _make_parser()
    _main(parser.parse_args())
