from pathlib import Path

from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class StanfordDogs(Dataset):
    def __init__(
        self, root, split="test", transform=None, loader=default_loader
    ) -> None:
        super().__init__()

        self.root = Path(root).joinpath("stanford_dogs")
        self.split = split
        self.transform = transform
        self.loader = loader

        assert self.split in ["train", "test"]

        list_path = self.root.joinpath(f"{self.split}_list.mat")
        split_file_list = loadmat(list_path, squeeze_me=True)

        self.files = split_file_list["file_list"].tolist()
        self.labels = (split_file_list["labels"] - 1).tolist()

        self.class_folder_to_idx = dict()
        for filename, label in zip(self.files, self.labels):
            cls_folder = filename.split("/")[0]
            if cls_folder not in self.class_folder_to_idx:
                self.class_folder_to_idx[cls_folder] = label

        self.classname_to_idx = {
            k.split("-", maxsplit=1)[1]: v for k, v in self.class_folder_to_idx.items()
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filepath = self.root.joinpath("Images", self.files[idx])
        label = self.labels[idx]
        image = self.loader(filepath)
        if self.transform is not None:
            image = self.transform(image)

        return image, label
