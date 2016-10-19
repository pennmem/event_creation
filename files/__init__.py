import os

def mkdir(path):
    os.mkdir(path, 0755)

def makedirs(path, **kwargs):
    os.makedirs(path, 0755, **kwargs)

class open_with_perms():

    def __init__(self, filename, mode='r', *args, **kwargs):
        self.filename = filename
        self.mode = mode
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        self.file = open(self.filename, self.mode, *self.args, **self.kwargs)
        return self.file

    def __exit__(self, exception_type, exception_value, traceback):
        self.file.close()
        if self.mode == 'w':
            os.chmod(self.filename, 0644)
