import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import torch.utils.checkpoint
from torch.utils.data import Dataset
import pytorch_lightning as L

from .sd import StableDiffusion
from .multiplier import Multiplier

from typing import Tuple

import itertools

import math

class SLiME(L.LightningModule):
    def __init__(
            self,
            pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2", # ["stabilityai/stable-diffusion-2", "stabilityai/stable-diffusion-2-base", "CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5"]

            train_timestep_range: Tuple[int, int] = (5, 100),
            test_timestep: int = 100,

            classes: int = 1,
            initial_name: str = "part",

            lr: float = 0.01,
            multiplier_lr: float = 0.5,

            alpha: float = 1.0,
            beta: float = 0.005,
    ):

        super().__init__()

        self.train_timestep_range = train_timestep_range
        self.test_timestep = test_timestep

        self.lr = lr
        self.multiplier_lr = multiplier_lr

        self.alpha = alpha
        self.beta = beta

        self.sd = StableDiffusion(pretrained_model_name_or_path)

        # freeze all sd parameters
        for param in self.sd.parameters():
            param.requires_grad = False

        [bos_token,initial_token,eos_token] = self.sd.tokenizer(initial_name).input_ids


        token_embeds = self.sd.text_encoder.get_input_embeddings().weight.data

        [bos_embedding,initial_embedding,eos_embedding] = token_embeds[[bos_token,initial_token,eos_token]]
        assert len(initial_embedding.shape) == 1, f"Initial embedding must be a vector, got shape {initial_embedding.shape}"
        embedding_dim = initial_embedding.shape[0]

        self.classes = classes
        self.text_tokens = classes + 1 # a background class + classes

        self.input_text_embeds = torch.zeros((1 + self.text_tokens + 1,embedding_dim))
        self.input_text_embeds[0] = bos_embedding
        # leave the middle embeddings as 0 -- these will be a learnable parameter
        self.input_text_embeds[self.text_tokens+1] = eos_embedding

        cls_text_embeds = torch.zeros((self.text_tokens,embedding_dim))
        cls_text_embeds[:] = token_embeds[initial_token]

        self.cls_text_embeds = nn.Parameter(cls_text_embeds)

        # Multipliers

        num_crosses,num_selfs = self.sd.num_attention_maps()

        self.cross_layer_multiplier = Multiplier(num_crosses)
        self.self_layer_multiplier = Multiplier(num_selfs)

        self.cross_map_multiplier = Multiplier(self.text_tokens)
        self.pred_map_multiplier = Multiplier(self.text_tokens)

    def mean_across_heads(self,mat,bsz):
        # convert (bh,*_) to (b,h,*_) then sum across h dimension
        bh,*rest = mat.shape
        return mat.view(bsz,-1,*rest).mean(1)

    def attn_maps_to_mask(
            self,
            cross_attn_maps,
            self_attn_maps,
            bsz,gt_dims,
        ):
        gt_tokens = gt_dims[0] * gt_dims[1]

        # cross_attn_maps = [cross_attn_map[1:-1] for cross_attn_map in cross_attn_maps] # remove bos and eos tokens
        # assert len(cross_attn_maps[0]) == self.text_tokens, f"Expected {self.text_tokens} cross maps, got {len(cross_attn_maps[0])}"

        # convert from many heads to one
        unified_cross_maps = [self.mean_across_heads(map,bsz).permute((0,2,1))[:,1:-1,:] for map in cross_attn_maps]# B,txt_tokens,im_tokens
        unified_self_maps = [self.mean_across_heads(map,bsz) for map in self_attn_maps]

        normed_cross_maps = [map/map.norm(dim=-1,keepdim=True) for map in unified_cross_maps] # normalize rows
        normed_self_maps = [map/map.norm(dim=-2,keepdim=True) for map in unified_self_maps] # normalize cols

        mean_cross_maps = torch.zeros((bsz,self.text_tokens,gt_tokens),device=self.device)
        for i,map in enumerate(normed_cross_maps):
            _,_,im_tokens = map.shape
            im_dim = int(math.sqrt(im_tokens))

            resized = F.interpolate(map.view(bsz,self.text_tokens,im_dim,im_dim),size=gt_dims,mode='bicubic')
            final = resized.view(bsz,self.text_tokens,gt_tokens)

            # TODO simplify this - maybe remove norm?
            scale = final.norm(dim=-1,keepdim=True) * len(unified_cross_maps) / torch.exp(self.cross_layer_multiplier.weight[i])
            final = final / scale

            mean_cross_maps += final


        mean_self_maps = torch.zeros((bsz,gt_tokens,gt_tokens),device=self.device)
        for i,map in enumerate(normed_self_maps):
            _,im_tokens,_ = map.shape
            im_dim = int(math.sqrt(im_tokens))

            # This is a cross attn map from pixels in space A -> pixels in space B. We resize both A and B to our target size
            resized = F.interpolate(map.view(bsz,im_tokens,im_dim,im_dim),size=gt_dims,mode='nearest')
            transposed = resized.view(bsz,im_tokens,gt_tokens).permute(0,2,1)
            t_resized = F.interpolate(transposed.view(bsz,gt_tokens,im_dim,im_dim),size=gt_dims,mode='nearest')

            final = t_resized.view(bsz,gt_tokens,gt_tokens).permute(0,2,1)

            scale = final.norm(dim=-2,keepdim=True) * len(unified_self_maps) / torch.exp(self.self_layer_multiplier.weight[i])
            final = final / scale

            mean_self_maps += final


        reshaped_cross_maps = mean_cross_maps.view(bsz,self.text_tokens,*gt_dims).permute(0,2,3,1)
        was_maps = torch.bmm(mean_cross_maps,mean_self_maps).view((bsz,self.text_tokens,*gt_dims)).permute(0,2,3,1) / gt_tokens

        cross_preds = reshaped_cross_maps.view(bsz,-1,self.text_tokens)
        self_preds = was_maps.view(bsz,-1,self.text_tokens)

        cross_preds = self.cross_map_multiplier(cross_preds)
        self_preds = self.pred_map_multiplier(self_preds)

        preds = cross_preds + self_preds

        return preds # shape (bsz,gt_tokens,self.text_tokens)
    
    def loss(
            self,
            sd_loss,
            pred,
            gt_masks,
            gt_masks_oh,
        ):

        bsz,*gt_dims = gt_masks.shape
        print(gt_masks.shape)

        assert self.classes == 1, f"Loss is only implemented for 1 class right now, got {self.classes}"

        targets = gt_masks.view(bsz,-1).float()

        # TODO: switch this to cross_entropy
        ce_loss = F.binary_cross_entropy_with_logits(pred[:,:,1],targets)
        mse_loss = F.mse_loss(pred,gt_masks_oh)

        loss = ce_loss + self.alpha * mse_loss + self.beta * sd_loss

        self.log_dict({
            "loss": loss,
            "mse_loss": mse_loss,
            "sd_loss": sd_loss,
            "ce_loss": ce_loss,
        })

        return loss


    def training_step(
            self,
            batch,
            batch_idx,
    ):

        pixel_values = batch["pixel_values"].to(self.device)

        gt_masks = batch["gt_masks"].to(self.device)
        gt_masks_oh = batch["gt_masks_oh"].to(self.device)

        bsz,*gt_dims = gt_masks.shape
        assert len(gt_dims) == 2, f"Expected 2D masks, got {gt_dims}"

        input_text_embeds = self.input_text_embeds.clone().to(self.device)
        input_text_embeds[1:self.text_tokens+1] = self.cls_text_embeds
        input_text_embeds = input_text_embeds.unsqueeze(0).expand(bsz,-1,-1)

        latents = self.sd.get_latents(pixel_values)
        # SLIME: sample timestamps from [5,100] as the paper recommends
        timesteps = self.train_timestep_range[0] + torch.randint(0, self.train_timestep_range[1] - self.train_timestep_range[0], (bsz,), device=self.device).long()

        sd_loss,cross_attn_maps,self_attn_maps = self.sd.training_step(
            latents,
            timesteps,
            input_text_embeds,
        )

        pred = self.attn_maps_to_mask(
            cross_attn_maps,
            self_attn_maps,
            bsz,gt_dims,
        )

        loss = self.loss(
            sd_loss,
            pred,
            gt_masks,
            gt_masks_oh,
        )

        return loss

    def predict_step(
            self,
            batch,
            batch_idx
    ):
        pixel_values = batch["pixel_values"].to(self.device)

        bsz,_,*gt_dims = pixel_values.shape

        input_text_embeds = self.input_text_embeds.clone()
        input_text_embeds[1:self.text_tokens+1] = self.cls_text_embeds
        input_text_embeds = input_text_embeds.unsqueeze(0).expand(bsz,-1,-1)

        latents = self.sd.get_latents(pixel_values)
        # SLIME: sample timestamps from [5,100] as the paper recommends
        timesteps = self.test_timestep * torch.ones((bsz,), device=self.device).long()

        _,cross_attn_maps,self_attn_maps = self.sd.training_step(
            latents,
            timesteps,
            input_text_embeds,
        )

        pred = self.attn_maps_to_mask(
            cross_attn_maps,
            self_attn_maps,
            bsz,gt_dims,
        )

        # return argmax of predictions
        return pred.argmax(-1)
    
    def configure_optimizers(self):
        return torch.optim.AdamW([
            {'params': [self.cls_text_embeds], 'lr': self.lr},
            {'params': itertools.chain(
                self.cross_layer_multiplier.parameters(),
                self.self_layer_multiplier.parameters(),

                self.cross_map_multiplier.parameters(),
                self.pred_map_multiplier.parameters(),
            ), 'lr': self.multiplier_lr}
        ])