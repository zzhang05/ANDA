# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from DiffAugment_pytorch import DiffAugment

#----------------------------------------------------------------------------
def shifted(img) :
    min_shift = 10
    max_shift = 25
    shift = min_shift + (max_shift - min_shift) * torch.rand(1)
    k = shift.long().item()
    new_data = img.clone()
    new_data[:,:,:k,:k] = img[:,:,-k:,-k:]
    new_data[:,:,:k,k:] = img[:,:,-k:,:-k]
    new_data[:,:,k:,:k] = img[:,:,:-k,-k:]
    return new_data

def get_perm(l) :
    perm = torch.randperm(l)
    while torch.all(torch.eq(perm, torch.arange(l))) :
        perm = torch.randperm(l)
    return perm

def jigsaw_k(data, k = 2) :
    with torch.no_grad() :
        actual_h = data.size()[2]
        actual_w = data.size()[3]
        h = torch.split(data, int(actual_h/k), dim = 2)
        splits = []
        for i in range(k) :
            splits += torch.split(h[i], int(actual_w/k), dim = 3)
        fake_samples = torch.stack(splits, -1)
        for idx in range(fake_samples.size()[0]) :
            perm = get_perm(k*k)
            # fake_samples[idx] = fake_samples[idx,:,:,:,torch.randperm(k*k)]
            fake_samples[idx] = fake_samples[idx,:,:,:,perm]
        fake_samples = torch.split(fake_samples, 1, dim=4)
        merged = []
        for i in range(k) :
            merged += [torch.cat(fake_samples[i*k:(i+1)*k], 2)]
        fake_samples = torch.squeeze(torch.cat(merged, 3), -1)
        return fake_samples

def stitch(data, k = 2) :
    #  = torch.randperm()
    indices = get_perm(data.size(0))
    data_perm = data[indices]
    actual_h = data.size()[2]
    actual_w = data.size()[3]
    if torch.randint(0, 2, (1,))[0].item() == 0 :
        dim0, dim1 = 2,3
    else :
        dim0, dim1 = 3,2

    h = torch.split(data, int(actual_h/k), dim = dim0)
    h_1 = torch.split(data_perm, int(actual_h/k), dim = dim0)
    splits = []
    for i in range(k) :
        if i < int(k/2) :
            splits += torch.split(h[i], int(actual_w/k), dim = dim1)
        else :
            splits += torch.split(h_1[i], int(actual_w/k), dim = dim1)
    merged = []
    for i in range(k) :
        merged += [torch.cat(splits[i*k:(i+1)*k], dim1)]
    fake_samples = torch.cat(merged, dim0)

    return fake_samples

def mixup(data, alpha = 25.0) :
    lamb = np.random.beta(alpha, alpha)
    # indices = torch.randperm(data.size(0))
    indices = get_perm(data.size(0))
    data_perm = data[indices]
    return data*lamb + (1-lamb)*data_perm

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutout(data) :
    min_k, max_k = 10, 20
    data = data.clone()
    h, w = data.size(2), data.size(3)
    b_size = data.size(0)
    for i in range(b_size) :
        k = (min_k + (max_k - min_k) * torch.rand(1)).long().item()
        h_pos = ((h - k) * torch.rand(1)).long().item()
        w_pos = ((w - k) * torch.rand(1)).long().item()
        patch = data[i,:,h_pos:h_pos+k,w_pos:w_pos+k]
        patch_mean = torch.mean(patch, axis = (1,2), keepdim = True)
        data[i,:,h_pos:h_pos+k,w_pos:w_pos+k] = torch.ones_like(patch) * patch_mean

    return data

def cut_mix(data, beta = 1.0) :
    data = data.clone()
    lam = np.random.beta(beta, beta)
    indices = get_perm(data.size(0))
    data_perm = data[indices]
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data_perm[:, :, bbx1:bbx2, bby1:bby2]
    return data

def rotate(data, angle = 60) :
    batch_size = data.size(0)
    new_data = []
    for i in range(batch_size) :
        pil_img = transforms.ToPILImage()(data[i].cpu())
        img_rotated = transforms.functional.rotate(pil_img, angle)
        new_data.append(transforms.ToTensor()(img_rotated))
    new_data = torch.stack(new_data, 0)
    return new_data.cuda()


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, diffaugment='', augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, with_dataaug=False):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.diffaugment = diffaugment
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.with_dataaug = with_dataaug
        self.pseudo_data = None

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        # Enable standard data augmentations when --with-dataaug=True
        if self.diffaugment:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def adaptive_negative_augmentation(self, gen_img):
        batch_size = gen_img.shape[0]
        pseudo_flag = torch.ones([batch_size, 1, 1, 1], device=self.device)
        pseudo_flag = torch.where(torch.rand([batch_size, 1, 1, 1], device=self.device) < self.augment_pipe.p,
                                  pseudo_flag, torch.zeros_like(pseudo_flag))
        if torch.allclose(pseudo_flag, torch.zeros_like(pseudo_flag)):
            assert self.pseudo_data is not None
            return 0.2 * self.pseudo_data * (1 - pseudo_flag) + 0.8 * gen_img * (1 + 0.25 * pseudo_flag)
        else:            
            return gen_img

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                # Update pseudo data
                self.pseudo_data = jigsaw_k(real_img, k=2)
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                if self.augment_pipe is not None:
                    gen_img_tmp = self.adaptive_negative_augmentation(gen_img)
                gen_logits = self.run_D(gen_img_tmp, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                # Apply Adaptive Pseudo Augmentation (APA) when --aug!='noaug'
                if self.diffaugment:
                    real_img_tmp = real_img
                else:
                    real_img_tmp = real_img
                real_img_tmp = real_img_tmp.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
