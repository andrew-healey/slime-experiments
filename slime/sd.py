import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import torch.utils.checkpoint
from torch.utils.data import Dataset
import pytorch_lightning as L

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from .attn_processor import CustomAttnProcessor

from typing import Tuple

class StableDiffusion(L.LightningModule):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            timestep_range: Tuple[int, int] = (5, 100),
        ):

        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )

        self.timestamp_range = timestep_range

        # Add the placeholder token in tokenizer
        num_added_tokens = self.tokenizer.add_tokens(["[0]","[1]"])
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the tokens."
            )
        self.cls_token_ids = [len(self.tokenizer)-2,len(self.tokenizer)-1]


        # Load models and create wrapper for stable diffusion
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )
        # Keep vae in eval mode as we don't train it
        self.vae.eval()

        # SLIME: we modify this UNet2DConditionModel
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        )
        # Keep unet in train mode to enable gradient checkpointing
        self.unet.train()

        self.__cross_attn_maps = []
        self.__self_attn_maps = []

        blocks = [
            *self.unet.down_blocks,
            self.unet.mid_block,
            *self.unet.up_blocks
        ]

        for block in blocks:
            if hasattr(block,"attentions"):
                for attn in block.attentions:
                    cross_processor = CustomAttnProcessor(self.__cross_attn_maps, len(self.__cross_attn_maps))
                    self_processor = CustomAttnProcessor(self.__self_attn_maps, len(self.__self_attn_maps))

                    self.__cross_attn_maps.append(None)
                    self.__self_attn_maps.append(None)

                    transformer_block = attn.transformer_blocks[0]
                    transformer_block.attn2.set_processor(cross_processor)
                    transformer_block.attn1.set_processor(self_processor)


        self.noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")
                    
    def clear_attn_maps(self):
        for i in range(len(self.__cross_attn_maps)):
            self.__cross_attn_maps[i] = None
        for i in range(len(self.__self_attn_maps)):
            self.__self_attn_maps[i] = None        

    def num_attention_maps(self):
        return len(self.__cross_attn_maps), len(self.__self_attn_maps)

    @torch.no_grad()
    def get_latents(self, pixel_values):
        # Convert images to latent space
        latents = self.vae.encode(pixel_values).latent_dist.sample().detach()
        latents = latents * 0.18215

        return latents


    def training_step(
            self,
            latents,
            timesteps,
            text_embeds,
        ):
        with torch.no_grad():
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_embeds

        # Predict the noise residual
        # SLIME: we extract CA and SA from Unet
        # cross_maps and self_maps are in-scope, use them

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()

        cross_attn_maps = list(self.__cross_attn_maps)
        self_attn_maps = list(self.__self_attn_maps)
        self.clear_attn_maps()

        return loss, cross_attn_maps, self_attn_maps