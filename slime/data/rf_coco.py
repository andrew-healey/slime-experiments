import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import torch.utils.checkpoint
from torch.utils.data import Dataset
import pytorch_lightning as L
import PIL
from PIL import Image

import numpy as np
from numpy import ndarray

import supervision as sv
from supervision import DetectionDataset,Detections

from typing import Tuple,Union

def load_dataset(rf_dataset:Union[Dataset,str]) -> Tuple[DetectionDataset,DetectionDataset]:
    dataset_location = rf_dataset.location if isinstance(rf_dataset,Dataset) else rf_dataset
    assert type(dataset_location) == str,f"dataset_location: {dataset_location}, type: {type(dataset_location)}"

    train_dataset = sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset_location}/train",
        annotations_path=f"{dataset_location}/train/_annotations.coco.json",
        force_masks=True
    )

    return train_dataset

class SemanticSegmentationDataset(Dataset):
    def __init__(
            self,
            detection_dataset: DetectionDataset,
            size=512,
            mask_size=64,
            interpolation="bicubic",
    ):
        self.detection_dataset = detection_dataset
        self.images = list(self.detection_dataset.images.values())
        self.size = size
        self.mask_size = mask_size

        self.num_images = len(self.images)
        self.num_classes = len(self.detection_dataset.classes)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL.Image.BILINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, i):
        img_name = self.images[i % self.num_images]
        image:ndarray = self.detection_dataset.images[img_name]
        detections:Detections = self.detection_dataset.annotations[img_name]

        
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example = {}
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        mask_torch_oh = torch.zeros((self.num_classes,self.size,self.size),dtype=bool)
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            mask_np = detections.mask[i]

            # resize mask, convert to torch, and convert to one-hot
            mask = Image.fromarray(mask_np)
            mask = mask.resize((self.mask_size,self.mask_size),resample=self.interpolation)
            mask_torch_oh[class_id] |= (TVF.pil_to_tensor(mask)[0] > 0).to(torch.int64)

        example["gt_masks_oh"] = mask_torch_oh
        return example
