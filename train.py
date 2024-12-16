import argparse
import os
import numpy as np
import datetime

from torch.utils.data import DataLoader

from model.ResGAN import GeneratorResNet, weights_init_normal, Discriminator
from per_loss import Per_Loss
from dataset import create_dataset
# from model.unet_model import UNet
from utils import LambdaLR, ReplayBuffer
from train_epoch import train_one_epoch, test_one_epoch
import torch.nn as nn
import torch.nn.functional as F
import torch

time_now = datetime.datetime.now().strftime('%m%d%H%M')
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--pretrained", type=str, default=None, help="pretrained model")
parser.add_argument("--batch_size", type=int, default=3, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=200, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--dataset_name", type=str, default='rain800', help="dataset name")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--lambda_gan", type=float, default=0.5, help="gan loss weight")
parser.add_argument("--lambda_per", type=float, default=0.1, help="con loss weight")
# parser.add_argument("--lambda_ff", type=float, default=1, help="ff loss weight")
parser.add_argument("--save_path", type=str, default='saved_models', help="model path")
parser.add_argument("--min_num", type=int, default=3, help='the number of model')
parser.add_argument("--data_root", type=str, default='dataset', help="data root")
parser.add_argument("--cuda_id", type=str, default='0', help="cuda number")
parser.add_argument("--lambda_id", type=list, default=[1.0, 0.5], help="cuda number")
opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_id

def main():
    results_file = opt.dataset_name + '_' + time_now + '.txt'
    # Losses
    criterion_GAN = torch.nn.BCELoss()
    criterion_identity = torch.nn.L1Loss()
    per_loss = Per_Loss()

    cuda = torch.cuda.is_available()
    # cuda = False
    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # Initialize generator and discriminator
    G = GeneratorResNet(input_shape, opt.n_residual_blocks)
    D = Discriminator(input_shape)
    if cuda:
        G = G.cuda()
        D = D.cuda()
        # coder = coder.cuda()
        criterion_GAN.cuda()
        criterion_identity.cuda()
        per_loss.cuda()
        print('use cuda')

    # Optimizers
    optimizer_G = torch.optim.Adam(
        G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # 1e-7
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_buffer = ReplayBuffer()

    # Training data loader
    dataloader = DataLoader(
        create_dataset(opt, opt.data_root),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu
    )
    # Test data loader
    val_dataloader = DataLoader(
        create_dataset(opt, opt.data_root, mode="test"),
        batch_size=1,
        shuffle=False
    )

    if opt.pretrained is not None:
        # Load pretrained models
        checkpoint = torch.load(os.path.join(opt.save_path, opt.dataset_name, opt.pretrained + ".pth"), map_location=torch.device('cpu'))
        G.load_state_dict(checkpoint['G_state'])
        D.load_state_dict(checkpoint['D_state'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        epoch = checkpoint['epoch']
        start_epoch = epoch + 1
        best_PSNR_SSIM = checkpoint['best_PSNR_SSIM']
        psnr_ssim_path_old = checkpoint['psnr_ssim_path_old']
        results_file = checkpoint['results_file']
    else:
        # Initialize weights
        G.apply(weights_init_normal)
        D.apply(weights_init_normal)
        start_epoch = opt.epoch
        best_PSNR_SSIM = [0.0, 0.0]
        psnr_ssim_path_old = []
        if not os.path.exists(os.path.join(opt.save_path, opt.dataset_name)):
            os.mkdir(os.path.join(opt.save_path, opt.dataset_name))

    for epoch in range(start_epoch, opt.n_epochs):
        print('epoch:', epoch)
        G, D, optimizer_G, optimizer_D, train_loss = \
            train_one_epoch(opt, dataloader, G, D, criterion_identity, criterion_GAN, per_loss,
                            optimizer_G, optimizer_D, Tensor, epoch, fake_buffer)
        torch.cuda.empty_cache()
        print('test:')
        with torch.no_grad():
            cur_PSNR, cur_SSIM = test_one_epoch(opt, val_dataloader, G, D,
                                                           criterion_identity, criterion_GAN, per_loss, Tensor,
                                                           epoch, fake_buffer)
        torch.cuda.empty_cache()
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], 0, cur_PSNR, cur_SSIM]
            txt = "epoch:{} ".format(epoch) + '  '.join('{:.4f}'.format(i) for i in result_info)
            print(txt)
            f.write(txt + "\n")

        # 保存最后一次模型
        checkpoint = {
            'G_state': G.state_dict(),
            'D_state': D.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'epoch': epoch,
            'best_PSNR_SSIM': best_PSNR_SSIM,
            'results_file': results_file,
            'psnr_ssim_path_old': psnr_ssim_path_old
        }
        torch.save(checkpoint, os.path.join(opt.save_path, opt.dataset_name, "last_{:d}.pth".format(epoch)))
        if epoch > 0:
            try:
                os.remove(os.path.join(opt.save_path, opt.dataset_name, "last_{:d}.pth".format(epoch - 1)))
            except:
                print('no model last')
        # 保存PSNR和SSIM组合精度最高的模型
        if cur_PSNR > best_PSNR_SSIM[0] and cur_SSIM > best_PSNR_SSIM[1]:
            best_PSNR_SSIM = [cur_PSNR, cur_SSIM]
            checkpoint = {
                'G_state': G.state_dict(),
                'D_state': D.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch,
                'best_PSNR_SSIM': best_PSNR_SSIM,
                'results_file': results_file,
                'psnr_ssim_path_old': psnr_ssim_path_old
            }
            torch.save(checkpoint, os.path.join(opt.save_path, opt.dataset_name,
                                                "best_{:d}_{:.2f}_{:.3f}.pth".format(epoch, best_PSNR_SSIM[0],
                                                                                     best_PSNR_SSIM[1])))

            psnr_ssim_path_old.append(os.path.join(opt.save_path, opt.dataset_name,
                                                   "best_{:d}_{:.2f}_{:.3f}.pth".format(epoch, best_PSNR_SSIM[0],
                                                                                        best_PSNR_SSIM[1])))

            if len(psnr_ssim_path_old) > opt.min_num:
                try:
                    os.remove(psnr_ssim_path_old[0])
                except:
                    print('no model psnr_ssim')
                psnr_ssim_path_old.remove(psnr_ssim_path_old[0])


if __name__ == '__main__':
    main()
