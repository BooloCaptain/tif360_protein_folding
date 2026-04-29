try:
    import sidechainnet as scn
except Exception:
    scn = None


def load_sidechainnet_split(*args, **kwargs):
    if scn is None:
        raise RuntimeError('sidechainnet is not installed. Install it or mock this function for tests.')
    return scn.load(*args, **kwargs)


class SidechainNetWrapper:
    """Minimal wrapper that defers to SidechainNet when available.

    This placeholder avoids importing heavy dependencies during initial scaffold.
    """
    def __init__(self, split='casp12'):
        self.split = split

    def __len__(self):
        if scn is None:
            return 0
        data = load_sidechainnet_split(self.split)
        return len(data)

    def __getitem__(self, idx):
        if scn is None:
            raise RuntimeError('sidechainnet not installed')
        data = load_sidechainnet_split(self.split)
        return data[idx]
