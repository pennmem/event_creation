import setuptools

install_requirements = []
try:
    import yaml

    install_requirements = yaml.load(open('conda_environment.yml'))
    install_requirements = install_requirements['dependencies'][:-1] + install_requirements['dependencies'][-1]['pip']
    # Since they haven't been uploaded to PyPI yet:
    install_requirements.remove('ptsa')
    install_requirements.remove('bptools')

except ImportError:
    print("Could not load conda_environment.yml; some requirements may be missing")

setuptools.setup(
    name='event_creation',
    author='Penn Computational Memory Lab',
    packages=  setuptools.find_packages(),
    entry_points = {
        "console_scripts":["submit=event_creation:submit",
                           "split=event_creation:split"]
    },
    zip_safe = False,
    install_requires = install_requirements,
    package_data = {'':['*.yml'],
                  'event_creation.submission':['*.json'],
                  'event_creation.submission.readers':['nk*'],
                  'event_creation.neurorad':['brainshift/*.R']
                  }

)