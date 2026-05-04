import numpy as np
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

class MaxTokensBatchSampler(Sampler):
    def __init__(self, lengths, max_tokens=4096, megabatch_size=10000, shuffle=True):
        self.lengths = np.array(lengths)
        self.max_tokens = max_tokens
        self.megabatch_size = megabatch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(lengths))

    def __iter__(self):
        # The magic is right here: an infinite while loop inside the sampler
        while True:
            if self.shuffle:
                np.random.shuffle(self.indices)

            for i in range(0, len(self.indices), self.megabatch_size):
                mega_indices = self.indices[i : i + self.megabatch_size]
                # Fast numpy sorting by length
                mega_indices = mega_indices[np.argsort(self.lengths[mega_indices])]
                
                current_batch = []
                max_len = 0
                for idx in mega_indices:
                    l = self.lengths[idx]
                    if max(max_len, l) * (len(current_batch) + 1) > self.max_tokens:
                        if current_batch:
                            yield current_batch
                        current_batch = [int(idx)]
                        max_len = l
                    else:
                        current_batch.append(int(idx))
                        max_len = max(max_len, l)
                
                if current_batch:
                    yield current_batch

    def __len__(self):
        # Standard PyTorch training expects a finite length, even if __iter__ is infinite
        return sum(self.lengths) // self.max_tokens + 1