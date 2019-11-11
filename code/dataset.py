from torch.utils.data import Dataset

# More information here
# (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class)
# and here (https://pytorch.org/docs/stable/data.html#dataset-types)


class ImageDataset(Dataset):
    def __init__(self):
        raise NotImplementedError

    def __getitem__(idx):
        raise NotImplementedError

    def __len__():
        raise NotImplementedError


class ScambledImageDataset(Dataset):
    def __init__(self):
        raise NotImplementedError

    def __getitem__(idx):
        raise NotImplementedError

    def __len__():
        raise NotImplementedError
