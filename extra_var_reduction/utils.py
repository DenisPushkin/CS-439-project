import time

import torchvision.datasets as _datasets
import torchvision.transforms as _transforms


def elapsed(last_time=[time.time()]):
    """ Returns the time passed since elapsed() was last called. """
    current_time = time.time()
    diff = current_time - last_time[0]
    last_time[0] = current_time
    return diff


def sampler(data, batch_size):
    if len(data) % batch_size is not 0:
        raise ValueError("Batch size should be divisible by mini batch size")
    start = 0
    while True:
        yield data[start:start+batch_size]
        start += batch_size
        start %= len(data)
        

def load_mnist(_data_root='datasets'):
    trans = [_transforms.ToTensor()]
    trans = _transforms.Compose(trans)
    _data = _datasets.MNIST(_data_root, train=True, download=True,
                            transform=trans)
    return _data