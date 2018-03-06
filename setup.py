import setuptools
from event_creation import __version___



setuptools.setup(
    name='event_creation',
    author='Penn Computational Memory Lab',
    version=__version___,
    url='https://github.com/pennmem/event_creation',
    download_url='https://github.com/pennmem/event_creation',
    packages=  setuptools.find_packages(),
    entry_points = {
        "console_scripts":["submit=event_creation:submit",
                           "split=event_creation:split"]
    },
    zip_safe = False,

    package_data = {'':['*.yml'],
                  'event_creation.submission':['*.json'],
                  'event_creation.submission.readers':['nk*'],
                  'event_creation.neurorad':['brainshift/*.R']
                  }

)
