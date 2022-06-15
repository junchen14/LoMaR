# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# iRPS: https://github.com/microsoft/Cream/tree/main/iRPE
# --------------------------------------------------------

from functools import partial
from re import T
from typing import MutableMapping
from unittest.mock import patch

import torch
import torch.nn as nn


from vision_transformer_irpe import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed




class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.encoder_pred = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def sample_patch_index_single_window(self,x,patch_index, keep_ratio):
        N, H, W, D = x.shape
        x = x.view(N,H*W,D)


        noise = torch.rand(N,patch_index.shape[0], device=patch_index.device)  # noise in [0, 1]
        
        ids_shuffle = torch.argsort(noise,dim=1)  # ascend: small is keep, large is remove

        ids_keep = ids_shuffle[:,:keep_ratio]

        patch_keeps = patch_index[ids_keep]

        return patch_keeps

    def sample_patch_index(self,x,patch_index, keep_ratio):

        N, H, W, D = x.shape
        M,P = patch_index.shape
        patch_index = patch_index.unsqueeze(0).expand(N,M,P)


        noise = torch.rand(N,M,P, device=patch_index.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise,dim=-1)  # ascend: small is keep, large is remove


        ids_keep = ids_shuffle[:,:,:keep_ratio]

        patch_keeps = torch.gather(patch_index, -1, ids_keep)

        return patch_keeps

    def generate_window_patches(self,x,left,top, window_size, mask_ratio):
        N, H, W, D = x.shape
        window_number = left.shape[0]
        

        #  extract the windows based on the coordinates
        left = left.unsqueeze(-1).expand(window_number,window_size)
        top  = top.unsqueeze(-1).expand(window_number, window_size)


        row = torch.arange(0,window_size,device=x.device).unsqueeze(0).expand(window_number,window_size)+left
        column = torch.arange(0,window_size*W,W, device = x.device).unsqueeze(0).expand(window_number, window_size)+top*W
        

        in_window_mask_number = int(window_size*window_size*mask_ratio)  

        assert in_window_mask_number>=1
        in_window_patches =row.unsqueeze(1).expand(window_number,window_size,window_size)  + column.unsqueeze(-1).expand(left.shape[0],window_size,window_size)
        in_window_patches = in_window_patches.view(window_number,-1)



        # sample the masked patch ids
        ids_mask_in_window =self.sample_patch_index(x,in_window_patches,in_window_mask_number)


        patches_to_keep = in_window_patches.unsqueeze(0).expand(N, window_number,window_size* window_size)
        x = x.view(N,H*W,D).unsqueeze(0).repeat(window_number,1, 1,1).view(N*window_number,H*W,D)


        sorted_patch_to_keep,_ = torch.sort(patches_to_keep,dim=-1)
        sorted_patch_to_keep = sorted_patch_to_keep.view(N*window_number,-1)

        ids_mask_in_window = ids_mask_in_window.view(N*window_number, -1)

        # gather the masked patches
        x_masked = torch.gather(x, dim=1, index=sorted_patch_to_keep.unsqueeze(-1).repeat(1, 1, D)).clone()
        # indices for recontruction
        mask_indices = ((sorted_patch_to_keep.unsqueeze(-1)- ids_mask_in_window.unsqueeze(1))==0).sum(-1)==1


        # zero out the patches in mask
        x_masked[mask_indices]=self.mask_token

 
        return x_masked, sorted_patch_to_keep,mask_indices


    def forward_encoder(self, x, window_size, num_window, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        x = x.type(torch.float32)

        N, _, C = x.shape
        H = W = self.img_size // self.patch_size
        x= x.view(N,H,W,C)
    

        assert window_size<= H and window_size <=W

        # sample window coordinates
        rand_top_locations = torch.randperm(H-window_size+1,device=x.device)[:num_window]
        rand_left_locations = torch.randperm(W-window_size+1,device=x.device)[:num_window]

        # generate the sampled and mask patches from the small windows
        x, ids_restore,mask_indices = self.generate_window_patches(x, rand_left_locations, rand_top_locations, window_size, mask_ratio)
                
        # append the cls tokens at the begining
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.encoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x, mask_indices, ids_restore



    def forward_loss(self, imgs, pred, mask_indices,num_window,ids_restore):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        N,P,H = target.shape

        target = target.unsqueeze(0).repeat(num_window,1,1,1).view(-1,P,H)
       

        target = torch.gather(target,dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, target.shape[-1]))

    
   
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch


        loss = (loss * mask_indices).sum() / mask_indices.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, window_size=7, num_window=4,mask_ratio=0.8):
        pred, mask_indices, ids_restore = self.forward_encoder(imgs, window_size,num_window,mask_ratio)
        loss = self.forward_loss(imgs, pred, mask_indices,num_window,ids_restore)
        return loss, pred, mask_indices




def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge448_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=448,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge672_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=672,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge996_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=996,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge336_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=336,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_384_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=384,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_448_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=448,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def mae_vit_base_patch14_224_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=224,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch8_224_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=224,
        patch_size=8, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_base_patch8_224 = mae_vit_base_patch8_224_dec512d8b
mae_vit_base_patch14_224 = mae_vit_base_patch14_224_dec512d8b
mae_vit_base_patch16_384 = mae_vit_base_patch16_384_dec512d8b
mae_vit_base_patch16_448 = mae_vit_base_patch16_448_dec512d8b

mae_vit_huge336_patch14 = mae_vit_huge336_patch14_dec512d8b
mae_vit_huge448_patch14 = mae_vit_huge448_patch14_dec512d8b
mae_vit_huge672_patch14 = mae_vit_huge672_patch14_dec512d8b
mae_vit_huge996_patch14 =mae_vit_huge996_patch14_dec512d8b





#mae_vit_huge448_patch14 = mae_vit_huge448_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
