import os
from contextlib import contextmanager


def mkdir(path):
    """Make a new directory with the correct permissions."""
    os.mkdir(path, 0o755)


def makedirs(path, **kwargs):
    """Make a series of directories with the correct permissions."""
    os.makedirs(path, 0o755, **kwargs)


@contextmanager
def open_with_perms(filename, mode='r', *args, **kwargs):
    """Opens a file and sets permissions to ``0o644`` when in write mode.

    Parameters
    ----------
    filename : str
    mode : str
    args : list
        Arguments to pass to :meth:`File.open`.
    kwargs : dict
        Keyword arguments to pass to :meth:`File.open`.

    """
    with open(filename, mode, *args, **kwargs) as f:
        yield f

    if mode == 'w':
        os.chmod(filename, 0o644)
