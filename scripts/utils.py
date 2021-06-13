import time

from torch.utils.data import DataLoader


def elapsed(last_time=[time.time()]):
    """ Returns the time passed since elapsed() was last called. """
    current_time = time.time()
    diff = current_time - last_time[0]
    last_time[0] = current_time
    return diff


def get_sampler(dataset, batch_size, shuffle=True, num_workers=1, drop_last=True):
  dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, 
                          num_workers=num_workers, drop_last=drop_last)
  dataloader_iterator = iter(dataloader)
  def sampler():
    nonlocal dataloader_iterator
    try:
        data = next(dataloader_iterator) 
    except StopIteration:
        dataloader_iterator = iter(dataloader)
        data = next(dataloader_iterator) 
    return data
  return sampler