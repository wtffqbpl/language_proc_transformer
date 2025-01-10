#! coding: utf-8

import torch

# torch.utils.data
# At the heart of PyTorch data loading utility is the `torch.utils.data.DataLoader` class.
# It represents a Python iterable over a dataset, with support for
#   * map-style and iterable-style datasets,
#   * customizing data loading order,
#   * automatic batching,
#   * single- and multi-process data loading,
#   * automatic memory pinning.
# These options are configured by the constructor arguments of a `DataLoader`, which has signature:
# DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            worker_init_fn=None, *, prefetch_factor=2,
#            persistent_workers=False)
#
# A map-style dataset is one that implements the `__getitem__` and `__len__` protocols, and
# represents a map from (possibly non-integral) indices/keys to data samples.
# For example, such a dataset, when accessed with `dataset[idx]`, could read the idx-th image
# and its corresponding label from a folder on the disk.


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end >= start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    # An iterable-style dataset is an instance of a subclass of `IterableDataset` that
    # implements the `__iter__()` protocol, and represents an iterable over data samples.
    # This type of datasets is particularly suitable for cases where random reads are
    # expensive or even improbable, and where the batch size depends on the fetched data.
    # For example, such a dataset, when called `iter(dataset)`, could return a stream
    # of data reading from a database, a remote server, or even logs generated in real time.
    #
    # When using a `IterableDataset` with multi-process data loading, the same dataset object
    # is replicated on each worker process, and thus the replicas must be configured
    # differently to avoid duplicated data.
    def __iter__(self):
        return iter(range(self.start, self.end))


def test_dataset():
    # should give some set of data as range(3, 7), i.e., [3, 4, 5, 6].
    ds = MyIterableDataset(start=3, end=7)

    # Single-process loading
    print(list(torch.utils.data.DataLoader(ds, num_workers=0)))

    # Directly doing multi-process loading yields duplicate data
    print(list(torch.utils.data.DataLoader(ds, num_workers=2)))


if __name__ == "__main__":
    test_dataset()
