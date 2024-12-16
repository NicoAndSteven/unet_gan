import sys
import time
import datetime

import skimage
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# from metrics import ssim, psnr
import cv2
from torchvision.transforms import functional as FF
from utils import transform_convert


def train_one_epoch(opt, dataloader, G, D, criterion_identity, criterion_GAN, per_loss, optimizer_G, optimizer_D, Tensor, epoch, fake_buffer):
    tqds = tqdm(dataloader)
    loss_G_rain_sum = 0.0
    loss_G_D_sum = 0.0
    loss_D_rain_sum = 0.0
    loss_D_clear_sum = 0.0
    loss_per_sum = 0.0
    for i, batch in enumerate(tqds):
        # Set model input

        rain = Variable(batch["rain"].type(Tensor))
        clear_real = Variable(batch["clear"].type(Tensor))

        G.train()
        D.train()
        optimizer_G.zero_grad()

        valid = Variable(Tensor(np.ones((rain.size(0), *D.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((rain.size(0), *D.output_shape))), requires_grad=False)
        clear_fake = G(rain)

        # Identity loss
        # 真假图片的损失（生成器生成的假去雨图和真去雨图）
        loss_G_rain = criterion_identity(clear_fake, clear_real) * opt.lambda_id[0]

        # con loss 感知损失（假去雨图和真去雨图）
        
        per_loss1 = per_loss(clear_fake, clear_real, rain)

        # GAN loss GAN损失 希望被判别器判别错误 MSE实现
        loss_GAN = criterion_GAN(D(clear_fake), valid)

        # Total loss 生成器总损失 = 生成器雨图损失 + GAN损失 + 感知损失
        loss_G = loss_G_rain + opt.lambda_gan * loss_GAN + opt.lambda_per * per_loss1

        loss_G_rain_sum += loss_G_rain.item()
        # loss_G_rain_sum2 += loss_G_rain2.item()
        loss_G_D_sum += opt.lambda_gan * loss_GAN.item()
        loss_per_sum += opt.lambda_per * per_loss1.item()
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D(clear_real), valid)
        loss_fake = criterion_GAN(D(clear_fake.detach()), fake)
        # Total loss
        # 鉴别器损失
        loss_D = ((loss_real + loss_fake) / 2) * opt.lambda_gan
        loss_D_rain_sum += loss_fake.item()
        loss_D_clear_sum += loss_real.item()
        loss_D.backward()
        optimizer_D.step()
        # break
    loss_G_rain_mean = loss_G_rain_sum / (i + 1)
    loss_G_D_mean = loss_G_D_sum / (i + 1)
    loss_D_rain_mean = loss_D_rain_sum / (i + 1)
    loss_D_clear_mean = loss_D_clear_sum / (i + 1)
    loss_per_mean = loss_per_sum / (i + 1)
    loss = [loss_G_rain_mean, loss_G_D_mean, loss_D_rain_mean,
            loss_D_clear_mean, loss_per_mean]
    return G, D, optimizer_G, optimizer_D, loss


def test_one_epoch(opt, dataloader, G, D, criterion_identity, criterion_GAN, contrast_loss, Tensor, epoch, fake_buffer):
    img_multiple_of = 8
    PSNR = []
    SSIM = []
    G.eval()
    tqds = tqdm(dataloader)

    for i, batch in enumerate(tqds):
        # Set model input
        rain = Variable(batch["rain"].type(Tensor))
        clear_real = Variable(batch["clear"].type(Tensor))

        h, w = rain.shape[2], rain.shape[3]
        H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                (w + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0
        rain_pd = F.pad(rain, (0, padw, 0, padh), 'reflect')

        w = rain.shape[3]
        clear_fake_pd = G(rain_pd)

        clear_fake_pd = torch.clamp(clear_fake_pd, 0, 1)
        # Unpad the output
        clear_fake = clear_fake_pd[:, :, :h, :w]
        #cv2.imshow('1',clear_fake)
        # 得到YCbCr中的Y通道数据
        clear_fake_y = skimage.color.rgb2ycbcr(clear_fake.permute(0, 2, 3, 1).squeeze(0).cpu().numpy())[:, :, 0]/255.
        clear_real_y = skimage.color.rgb2ycbcr(clear_real.permute(0, 2, 3, 1).squeeze(0).cpu().numpy())[:, :, 0]/255.

        PSNR.append(psnr(clear_fake_y, clear_real_y, data_range=1.0))
        SSIM.append(ssim(clear_fake_y, clear_real_y, win_size=11, data_range=1.0, gaussian_weights=True))
        # break
    PSNR_mean = np.mean(PSNR)
    SSIM_mean = np.mean(SSIM)
    print(PSNR_mean, SSIM_mean)
    return PSNR_mean, SSIM_mean
