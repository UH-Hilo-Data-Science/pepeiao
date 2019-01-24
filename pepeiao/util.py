import collections
import csv
import logging
import pkg_resources


from pepeiao.constants import _BEGIN_KEY, _END_KEY, _RAVEN_HEADER

_LOGGER = logging.getLogger(__name__)

def in_interval(time, interval):
    """Is time in interval (start, end)?"""
    return (time >= interval[0]) and (time <= interval[1])

def load_selections(filename):
    """Parse a raven selections table and return a list of dictionaries"""
    with open(filename) as csvfile:
        reader = InsensitiveReader(csvfile, fieldnames=_RAVEN_HEADER, dialect='excel-tab')
        rows = [x for x in reader]
    _LOGGER.info('Read %d selections from %s.', len(rows), filename)
    return rows

def progress_dots(iterable, start=None, end=' Done.', char='.', stride=1):
    """Yield the items of an iterable, printing char to stdout as term is evaluated.

    Arguments:
    - iterable -- item to iterate over
    - start -- string to print at beginning of progress bar
    - end -- string to print when the iterable is exausted
    - char -- string to print as each item is evaluated
    - stride -- char will be printed every stride-th item
"""
    if start:
        print(start, end='', flush=True)
    for idx, item in enumerate(iterable, 1):
        if not idx % stride:
            print(char, end='', flush=True)
        yield item
    print(end, flush=True)

def selections_to_intervals(selections):
    """Extract (start, end) pairs from rows of a selection table."""
    return [(float(r[_BEGIN_KEY]), float(r[_END_KEY])) for r in selections
                if all(k in r for k in (_BEGIN_KEY, _END_KEY))]

def intervals_to_selections(intervals):
    pass


class InsensitiveReader(csv.DictReader):
    """A csv.DictReader that ignores captialization of field names."""
    @property
    def fieldnames(self):
        return [x.strip().lower() for x in super().fieldnames]

    def __next__(self):
        return InsensitiveDict(super().__next__())

    
class InsensitiveDict(collections.OrderedDict):
    """Case insensitive OrderedDict"""

    def __getitem__(self, key):
        return super().__getitem__(key.strip().lower())

    def __setitem__(self, key, value):
        try:
            key = key.strip().lower()
        except AttributeError:
            pass
        return super().__setitem__(key, value)

    def get(self, key, default=None):
        return super().get(key.strip().lower(), default)


def get_models():
    models = dict()
    for entry_point in pkg_resources.iter_entry_points('pepeiao_models'):
        new_models = entry_point.load()
        both = [k for k in new_models if k in models]
        if both:
            _LOGGER.warn('Pepeiao loading a model with identical name(s): %s'.format(both))
        models.update(new_models)
    return  models
