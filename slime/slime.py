import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torch.nn import InstanceNorm2d,BatchNorm2d
import torch.utils.checkpoint
from torch.utils.data import Dataset
import pytorch_lightning as L

from .sd import StableDiffusion
from .multiplier import Multiplier

from typing import Tuple, List

import itertools

import math
import time

def dice_loss(pred, target):
    smooth = 1.

    pred = torch.sigmoid(pred)

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

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
            gamma: float = 0.0,

            cross_attn_nums: List[int] = [8,9,10,11,12],
            self_attn_nums: List[int] = [14,15,16],

            use_self_attn: bool = True,
    ):

        super().__init__()

        self.train_timestep_range = train_timestep_range
        self.test_timestep = test_timestep

        self.lr = lr
        self.multiplier_lr = multiplier_lr

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.cross_attn_nums = cross_attn_nums
        self.self_attn_nums = self_attn_nums

        self.use_self_attn = use_self_attn

        self.sd = StableDiffusion(pretrained_model_name_or_path)

        self.normed_cross_maps= [] # for debugging

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

        del self.sd.text_encoder
        del self.sd.tokenizer

        # Multipliers

        num_crosses,num_selfs = self.sd.num_attention_maps()

        self.cross_layer_multiplier = Multiplier(num_crosses)
        self.self_layer_multiplier = Multiplier(num_selfs)

        self.cross_map_multiplier = Multiplier(self.text_tokens)
        self.pred_map_multiplier = Multiplier(self.text_tokens)

        self.cross_norm = BatchNorm2d(len(cross_attn_nums),affine=True)
        self.cross_map_norm = BatchNorm2d(self.text_tokens,affine=True)

        self.latest_xattns = []
        self.latest_means = []
        self.latest_preds = []

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

        # clear out unused

        for i in range(len(cross_attn_maps)):
            if i+1 not in self.cross_attn_nums:
                cross_attn_maps[i] = None
        for i in range(len(self_attn_maps)):
            if i+1 not in self.self_attn_nums:
                self_attn_maps[i] = None

        # convert from many heads to one
        unified_cross_maps = [self.mean_across_heads(map,bsz).permute((0,2,1)) for i,map in enumerate(cross_attn_maps) if i+1 in self.cross_attn_nums]# B,txt_tokens,im_tokens
        unified_self_maps = torch.stack([self.mean_across_heads(map,bsz) for i,map in enumerate(self_attn_maps) if i+1 in self.self_attn_nums])

        # normed_cross_maps = [map/map.norm(dim=-1,keepdim=True) for map in unified_cross_maps] # normalize rows
        # normed_self_maps = [map/map.norm(dim=-2,keepdim=True) for map in unified_self_maps] # normalize cols

        # self.normed_cross_maps = [tensor.detach().cpu() for tensor in normed_cross_maps]

        mean_cross_maps = torch.zeros((bsz,len(unified_cross_maps),self.text_tokens,gt_tokens),device=self.device)
        for i,map in enumerate(unified_cross_maps):

            map = map[:,1:-1]
            assert map.shape[1] == self.text_tokens

            _,_,im_tokens = map.shape
            im_dim = int(math.sqrt(im_tokens))

            reshaped_map = map.view(bsz,self.text_tokens,im_dim,im_dim)

            # make sure interpolation preserves the mean of the map
            scale_factor = gt_tokens / im_tokens
            resized = F.interpolate(reshaped_map,size=gt_dims,mode='bicubic')

            final = resized.view(bsz,self.text_tokens,gt_tokens)

            mean_cross_maps[:,i] = final * (1 * scale_factor) * self.cross_layer_multiplier.weight[i]

            # # TODO simplify this - maybe remove norm?
            # scale = torch.exp(self.cross_layer_multiplier.weight[i])
            # final = final / scale

            # mean_cross_maps +=
        
        self.latest_xattns = [map.cpu().detach().numpy() for map in cross_attn_maps if map is not None]
        self.latest_means = [map.cpu().detach().numpy() for map in mean_cross_maps if map is not None]
        mean_cross_map = mean_cross_maps.mean(dim=1)#self.cross_norm(mean_cross_maps).mean(dim=1)
        # import pdb; pdb.set_trace()

        # print(mean_cross_maps.mean())
        # import pdb; pdb.set_trace()

        del unified_cross_maps,mean_cross_maps

        mean_self_maps = torch.zeros((bsz,gt_tokens,gt_tokens),device=self.device)
        for i,map in enumerate(unified_self_maps):

            _,im_tokens,_ = map.shape
            im_dim = int(math.sqrt(im_tokens))

            # This is a cross attn map from pixels in space A -> pixels in space B. We resize both A and B to our target size
            resized = F.interpolate(map.view(bsz,im_tokens,im_dim,im_dim),size=gt_dims,mode='nearest')
            transposed = resized.view(bsz,im_tokens,gt_tokens).permute(0,2,1)
            t_resized = F.interpolate(transposed.view(bsz,gt_tokens,im_dim,im_dim),size=gt_dims,mode='nearest')

            final = t_resized.view(bsz,gt_tokens,gt_tokens).permute(0,2,1)

            scale = final.norm(dim=-2,keepdim=True) / torch.exp(self.self_layer_multiplier.weight[i])
            final = final / scale

            mean_self_maps += final
        
        del unified_self_maps

        reshaped_cross_maps = mean_cross_map.view(bsz,self.text_tokens,*gt_dims)
        # import pdb; pdb.set_trace()
        rescaled_maps = self.cross_map_norm(reshaped_cross_maps)
        cross_preds = rescaled_maps.permute(0,2,3,1).view(bsz,-1,self.text_tokens)
        # cross_preds = self.cross_map_multiplier(cross_preds)
        # import pdb;pdb.set_trace()
        # cross_preds = self.cross_norm(cross_preds)

        if self.use_self_attn:
            was_maps = torch.bmm(mean_cross_map,mean_self_maps).view((bsz,self.text_tokens,*gt_dims)).permute(0,2,3,1) / gt_tokens
            self_preds = was_maps.view(bsz,-1,self.text_tokens)
            self_preds = self.pred_map_multiplier(self_preds)

        preds = cross_preds + self_preds if self.use_self_attn else cross_preds

        # preds = preds[:,:,1:]
        assert preds.shape[2] == self.text_tokens,f"Preds shape is {preds.shape}"

        # import pdb; pdb.set_trace()

        self.latest_preds.append(preds.cpu().detach().numpy())
        self.latest_preds = self.latest_preds[-10:]

        return preds # shape (bsz,gt_tokens,self.text_tokens)
    
    def loss(
            self,
            sd_loss,
            pred,
            gt_masks,
            gt_masks_oh,
        ):

        bsz,*gt_dims,_ = gt_masks_oh.shape

        assert self.classes == 1, f"Loss is only implemented for 1 class right now, got {self.classes}"

        targets = gt_masks_oh.view(bsz,-1,self.classes).float()

        ce_loss = F.binary_cross_entropy_with_logits(pred,targets)
        mse_loss = F.mse_loss(torch.sigmoid(pred),gt_masks_oh.view((bsz,-1,self.classes)).float())
        dice = dice_loss(pred,gt_masks_oh.view((bsz,-1,self.classes)).float())

        print("pred", round(pred.min().cpu().item(), 2), round(pred.max().cpu().item(), 2), "ce loss", round(ce_loss.cpu().item(), 2), "mse loss", round(mse_loss.cpu().item(), 2), "dice loss", round(dice.cpu().item(), 2))

        # import pdb; pdb.set_trace()

        loss = ce_loss + self.alpha * mse_loss + self.beta * sd_loss + self.gamma * dice

        self.log_dict({
            "loss": loss,
            "mse_loss": mse_loss,
            "sd_loss": sd_loss,
            "ce_loss": ce_loss,
            "dice_loss": dice,
        })

        return loss


    def training_step(
            self,
            batch,
            batch_idx,
    ):
        start_time = time.time()

        pixel_values = batch["pixel_values"].to(self.device)

        gt_masks = None#batch["gt_masks"].to(self.device)
        gt_masks_oh = batch["gt_masks_oh"].to(self.device)

        bsz,*gt_dims,_ = gt_masks_oh.shape
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
            pred[:,:,1:],
            gt_masks,
            gt_masks_oh,
        )

        self.log("duration",time.time()-start_time)

        return loss

    @torch.no_grad()
    def predict_step(
            self,
            batch,
            batch_idx,
            gt_dim=64
    ):
        pixel_values = batch["pixel_values"].to(self.device)

        bsz,*_ = pixel_values.shape
        gt_dims = (gt_dim,gt_dim)

        input_text_embeds = self.input_text_embeds.clone().to(self.device)
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
        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': [self.cls_text_embeds], 'lr': self.lr},
            {'params': itertools.chain(
                self.cross_layer_multiplier.parameters(),
                self.self_layer_multiplier.parameters(),

                self.cross_map_multiplier.parameters(),
                self.pred_map_multiplier.parameters(),
                self.cross_norm.parameters(),
                self.cross_map_norm.parameters(),
            ), 'lr': self.multiplier_lr}
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=0,
        )
        return [optimizer], [scheduler]
