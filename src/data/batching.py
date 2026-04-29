import math
from torch.utils.data import Sampler


def bucket_by_length(lengths, batch_size):
    """Simple bucketing: sort indices by length and yield batches of indices.

    lengths: list or array of sequence lengths
    returns: list of lists of indices
    """
    idxs = list(range(len(lengths)))
    idxs.sort(key=lambda i: lengths[i])
    batches = [idxs[i:i+batch_size] for i in range(0, len(idxs), batch_size)]
    return batches


class BucketBatchSampler(Sampler):
    """Length-based batching to reduce padding overhead."""

    def __init__(self, lengths, batch_size):
        self.batches = bucket_by_length(lengths, batch_size)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
