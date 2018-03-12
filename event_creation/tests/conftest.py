import sys

from_test = False

this  = sys.modules[__name__]


def pytest_configure(config):
    this.from_test = True

def pytest_unconfigure(config):
    this.from_test = False

if __name__ == '__main__':
    pytest_configure(None)
    print(from_test)