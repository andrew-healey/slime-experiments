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
      use_augmentations=True,
  ):
    self.data_root = data_root
    self.mask_root = mask_root
    self.size = size
    self.mask_size=mask_size
    self.random_crop = random_crop
    self.horizontal_flip = horizontal_flip
    self.brightness_contrast_adjust = brightness_contrast_adjust
    self.num_augmentations = num_augmentations
    self.use_augmentations = use_augmentations

    self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
    self.image_paths.sort()
    if self.mask_root is not None:
      self.mask_paths = [os.path.join(self.mask_root, file_path) for file_path in os.listdir(self.mask_root)]
      self.mask_paths.sort()

      assert len(self.image_paths) == len(self.mask_paths),f"Number of images and masks must match. Got {len(self.image_paths)} images and {len(self.mask_paths)} masks."
      assert [os.path.basename(image_path).split(".")[0] == os.path.basename(mask_path).split(".")[0] for image_path,mask_path in zip(self.image_paths,self.mask_paths)],"Image and mask names must match."

    else:
      self.mask_paths = None

    self.num_images = len(self.image_paths)
    self._length = self.num_images * self.num_augmentations if self.use_augmentations else self.num_images

    self.interpolation = {
      "linear": PIL.Image.LINEAR,
      "bilinear": PIL.Image.BILINEAR,
      "bicubic": PIL.Image.BICUBIC,
      "lanczos": PIL.Image.LANCZOS,
    }[interpolation]

    self.aug = A.Compose([
      A.RandomResizedCrop(height=self.size, width=self.size) if self.random_crop else A.NoOp(),
      A.HorizontalFlip() if self.horizontal_flip else A.NoOp(),
      A.RandomBrightnessContrast() if self.brightness_contrast_adjust else A.NoOp(),
    ])

  def __len__(self):
    return self._length

  def __getitem__(self, i):
    example = {}
    img_path = self.image_paths[i % self.num_images]
    image = Image.open(img_path)

    if not image.mode == "RGB":
        image = image.convert("RGB")

    # default to score-sde preprocessing
    img = np.array(image).astype(np.uint8)

    if self.mask_root is not None:
      mask_path = self.mask_paths[i // self.num_augmentations % self.num_images]
      mask = Image.open(mask_path)
      mask = mask.resize(image.size, resample=self.interpolation)
      mask = np.array(mask).astype(np.uint8)
      if self.use_augmentations:
        augmented = self.aug(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

    image = Image.fromarray(img)
    image = image.resize((self.size, self.size), resample=self.interpolation)

    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)

    example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

    if self.mask_root is not None:
      mask = Image.fromarray(mask)
      mask = mask.resize((self.mask_size, self.mask_size), resample=self.interpolation)
      mask_torch = (TVF.pil_to_tensor(mask)[0] > 0)
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
