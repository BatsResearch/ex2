from pathlib import Path

import torchvision

import cub_dataset
import stanford_dogs_dataset


def get_vision_dataset(
    config,
    per_cls_indices=False,
    preprocess=None,
    num_images_per_cls=None,
):
    if config["dataset"] == "cub":
        cub_path = Path(config["root"], "cub")
        if not cub_path.exists():
            cub_path = Path(config["root"])

        split_ = config["split"]
        dataset = cub_dataset.Cub2011(
            root=cub_path.as_posix(),
            train=split_ == "train",
            transform=preprocess,
        )
        image_and_targets = list()
        for idx in range(len(dataset)):
            sample = dataset.data.iloc[idx]
            path = Path(dataset.root, dataset.base_folder, sample.filepath).as_posix()
            target = sample.target - 1
            image_and_targets.append((path, target))
    elif config["dataset"] == "fgvc_aircraft":
        split_ = config["split"]
        dataset = torchvision.datasets.FGVCAircraft(
            root=config["root"], transform=preprocess, split=split_
        )

        image_and_targets = [
            (
                Path(dataset._image_files[idx]).as_posix(),
                dataset._labels[idx],
            )
            for idx in range(len(dataset))
        ]
    elif config["dataset"] == "flowers":
        split_ = config["split"]
        dataset = torchvision.datasets.Flowers102(
            root=config["root"], transform=preprocess, split=split_
        )

        image_and_targets = [
            (Path(dataset._image_files[idx]).as_posix(), dataset._labels[idx])
            for idx in range(len(dataset))
        ]
    elif config["dataset"] == "pets":
        split_ = config["split"]
        dataset = torchvision.datasets.OxfordIIITPet(
            root=config["root"], transform=preprocess, split=split_
        )
        image_and_targets = [
            (Path(dataset._images[idx]).as_posix(), dataset._labels[idx])
            for idx in range(len(dataset))
        ]
    elif config["dataset"] == "stanford_cars":
        split_ = config["split"]
        dataset = torchvision.datasets.StanfordCars(
            root=config["root"], transform=preprocess, split=split_
        )
        image_and_targets = [
            (
                Path(dataset._samples[idx][0]).as_posix(),
                dataset._samples[idx][1],
            )
            for idx in range(len(dataset))
        ]
    elif config["dataset"] == "stanford_dogs":
        split_ = config["split"]
        dataset = stanford_dogs_dataset.StanfordDogs(
            root=config["root"], transform=preprocess, split=split_
        )

        image_and_targets = [
            (
                Path(dataset.root.joinpath("Images", dataset.files[idx])).as_posix(),
                dataset.labels[idx],
            )
            for idx in range(len(dataset))
        ]
    else:
        raise ValueError

    per_cls_img_indices = None
    if per_cls_indices:
        file2idx = dict()
        per_cls_files = dict()

        for rec_idx in range(len(dataset)):
            image_path, label = image_and_targets[rec_idx]
            file2idx[image_path] = rec_idx
            if label not in per_cls_files:
                per_cls_files[label] = list()
            per_cls_files[label].append(image_path)

        for k in range(len(per_cls_files)):
            per_cls_files[k] = list(sorted(per_cls_files[k]))

        per_cls_img_indices = dict()
        for k in per_cls_files.keys():
            if (
                num_images_per_cls is not None
                and num_images_per_cls != -1
                and num_images_per_cls != 0
            ):
                chosen_files = per_cls_files[k][:num_images_per_cls]
            else:
                chosen_files = per_cls_files[k]

            per_cls_img_indices[k] = [file2idx[f] for f in chosen_files]

    return dataset, per_cls_img_indices
