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

"""
Generic binary segmentation dataset - performs semantic segmentation on a single class.
"""
import albumentations as A

class BinarySegmentationDataset(Dataset):

  def __init__(
      self,
      data_root,
      mask_root,
      size=512,
      mask_size=64,
      interpolation="bicubic",
      random_crop=True,
      horizontal_flip=True,
      brightness_contrast_adjust=True,
      num_augmentations=5,
  ):
    self.data_root = data_root
    self.mask_root = mask_root
    self.size = size
    self.mask_size=mask_size
    self.random_crop = random_crop
    self.horizontal_flip = horizontal_flip
    self.brightness_contrast_adjust = brightness_contrast_adjust
    self.num_augmentations = num_augmentations

    self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
    if self.mask_root is not None:
      self.mask_paths = [os.path.join(self.mask_root, file_path) for file_path in os.listdir(self.mask_root)]
    else:
      self.mask_paths = None

    self.num_images = len(self.image_paths)
    self._length = self.num_images * self.num_augmentations

    self.interpolation = {
      "linear": PIL.Image.LINEAR,
      "bilinear": PIL.Image.BILINEAR,
      "bicubic": PIL.Image.BICUBIC,
      "lanczos": PIL.Image.LANCZOS,
    }[interpolation]

    self.aug = A.Compose([
      A.RandomCrop(width=self.size, height=self.size) if self.random_crop else A.NoOp(),
      A.HorizontalFlip() if self.horizontal_flip else A.NoOp(),
      A.RandomBrightnessContrast() if self.brightness_contrast_adjust else A.NoOp(),
    ])

  def __len__(self):
    return self._length

  def __getitem__(self, i):
    example = {}
    image = Image.open(self.image_paths[i // self.num_augmentations % self.num_images])

    if not image.mode == "RGB":
        image = image.convert("RGB")

    # default to score-sde preprocessing
    img = np.array(image).astype(np.uint8)

    img = self.aug(image=img)['image']

    image = Image.fromarray(img)
    image = image.resize((self.size, self.size), resample=self.interpolation)

    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)

    example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

    if self.mask_root is not None:
      mask = Image.open(self.mask_paths[i // self.num_augmentations % self.num_images])
      mask_torch = (TVF.pil_to_tensor(mask.resize((self.mask_size,self.mask_size),resample=self.interpolation))[0] > 0)
      mask_torch_oh = mask_torch[...,None]
      example["gt_masks_oh"] = mask_torch_oh
      # mask_torch_oh = F.one_hot(mask_torch,num_classes=2) # hardcode to (background,foreground)
    else:
      mask_torch_oh = None

    return example


class SegmentationDataModule(L.LightningDataModule):
    batch_size: int = 2
    iters_per_epoch: int = 50

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset=None,
        iters_per_epoch:int=50,
        batch_size:int=2,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.iters_per_epoch = iters_per_epoch
        self.batch_size = batch_size
    
    def cycle(self,iterable,max_iters):
      iters = 0
      while iters < max_iters:
          for x in iterable:
              yield x
              iters+=1

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )
        return self.cycle(loader,self.iters_per_epoch)

    def test_dataloader(self):
      if self.test_dataset is None:
        return None
      return torch.utils.data.DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          pin_memory=True,
          drop_last=False,
      )
