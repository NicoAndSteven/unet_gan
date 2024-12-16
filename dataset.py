import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as FF
import torch.nn.functional as F


def create_dataset(opt, root, transforms=None, mode="train", test_name='Rain100H'):
    if opt.dataset_name == 'ship':
        return ship_ImageDataset(opt, root=root, transforms=transforms, mode=mode)
    elif opt.dataset_name == 'rain800':
        return rain800_ImageDataset(opt, root=root, transforms=transforms, mode=mode)

    elif opt.dataset_name == 'Lv2000U':
        return Lv2000U_ImageDataset(opt, root=root, transforms=transforms, mode=mode)
    

    elif opt.dataset_name == 'rain1200':
        return DID_ImageDataset(opt, root=root, transforms=transforms, mode=mode)
    elif opt.dataset_name == 'rain1400':
        return rain1400_ImageDataset(opt, root=root, transforms=transforms, mode=mode)
    elif opt.dataset_name == 'rain200h':
        return rain200_ImageDataset(opt, root=root, transforms=transforms, mode=mode, name='Rain200H')
    elif opt.dataset_name == 'rain200l':
        return rain200_ImageDataset(opt, root=root, transforms=transforms, mode=mode, name='Rain200L')
    elif opt.dataset_name == 'rain100h':
        return rain100_ImageDataset(opt, root=root, transforms=transforms, mode=mode, name='Rain100H')
    elif opt.dataset_name == 'rain100l':
        return rain100_ImageDataset(opt, root=root, transforms=transforms, mode=mode, name='Rain100L')
    elif opt.dataset_name == 'rain13k':
        return rain13k_ImageDataset(opt, root=root, transforms=transforms, mode=mode, test_name=test_name)
    else:
        print('error dataset name')
        exit()


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class Lv2000U_ImageDataset(Dataset):
    def __init__(self, opt, root, transforms=None, mode="train"):
        self.transform = transforms
        self.size = opt.img_height
        self.mode = mode
        self.root = os.path.join(root, 'Lv2000U')
        if mode == 'train':
            with open(os.path.join(self.root, 'train.txt'), 'r') as f:
                self.img_path = [os.path.join(self.root, 'training', line.strip())
                                 for line in f.readlines() if len(line.strip()) > 0]
        else:
            with open(os.path.join(self.root, 'test.txt'), 'r') as f:
                self.img_path = [os.path.join(self.root, 'Lv2000U_test', line.strip())
                                 for line in f.readlines() if len(line.strip()) > 0]

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        w, h = img.size
        # Convert grayscale images to rgb
        if img.mode != "RGB":
            img = to_rgb(img)

        clear_img = img.crop((0, 0, w / 2, h))
        rain_img = img.crop((w / 2, 0, w, h))

        if self.mode == 'train':
            item_rain, item_clear = train_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        else:
            item_rain, item_clear = val_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        return {"rain": item_rain, "clear": item_clear}

    def __len__(self):
        return len(self.img_path)


class ship_ImageDataset(Dataset):
    def __init__(self, opt, root, transforms=None, mode="train"):
        self.transform = transforms
        self.size = opt.img_height
        self.mode = mode
        self.root = os.path.join(root, 'ship_new')
        with open(os.path.join(self.root, mode + '.txt'), 'r') as f:
            self.clear_path = [os.path.join(self.root, 'clear', line.strip())
                               for line in f.readlines() if len(line.strip()) > 0]
        with open(os.path.join(self.root, mode + '.txt'), 'r') as f:
            self.rain_path = [os.path.join(self.root, 'rain_new', line.strip())
                              for line in f.readlines() if len(line.strip()) > 0]

    def __getitem__(self, index):
        rain_img = Image.open(self.rain_path[index] + '.jpg')
        clear_img = Image.open(self.clear_path[index] + '.jpg')
        if rain_img.size != clear_img.size:
            print(1)
        # Convert grayscale images to rgb
        if rain_img.mode != "RGB":
            rain_img = to_rgb(rain_img)
        if clear_img.mode != "RGB":
            clear_img = to_rgb(clear_img)

        if self.mode == 'train':
            item_rain, item_clear = train_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        else:
            item_rain, item_clear = val_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
            item_rain = transforms.CenterCrop((512, 512))(item_rain)
            item_clear = transforms.CenterCrop((512, 512))(item_clear)
        return {"rain": item_rain, "clear": item_clear}

    def __len__(self):
        return len(self.rain_path)


class rain800_ImageDataset(Dataset):
    def __init__(self, opt, root, transforms=None, mode="train"):
        self.transform = transforms
        self.size = opt.img_height
        self.mode = mode
        self.root = os.path.join(root, 'Rain800')
        if mode == 'train':
            with open(os.path.join(self.root, 'train.txt'), 'r') as f:
                self.img_path = [os.path.join(self.root, 'training', line.strip())
                                 for line in f.readlines() if len(line.strip()) > 0]
        else:
            with open(os.path.join(self.root, 'test_no60_71.txt'), 'r') as f:
                self.img_path = [os.path.join(self.root, 'rain800_test', line.strip())
                                 for line in f.readlines() if len(line.strip()) > 0]

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        w, h = img.size
        # Convert grayscale images to rgb
        if img.mode != "RGB":
            img = to_rgb(img)

        clear_img = img.crop((0, 0, w / 2, h))
        rain_img = img.crop((w / 2, 0, w, h))

        if self.mode == 'train':
            item_rain, item_clear = train_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        else:
            item_rain, item_clear = val_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        return {"rain": item_rain, "clear": item_clear}

    def __len__(self):
        return len(self.img_path)


class DID_ImageDataset(Dataset):
    def __init__(self, opt, root, transforms=None, mode="train"):
        self.transform = transforms
        self.size = opt.img_height
        self.root = os.path.join(root, 'DID-MDN-datasets')
        self.mode = mode
        with open(os.path.join(self.root, mode + '.txt'), 'r') as f:
            self.img_name = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
        self.sig = len(self.img_name)
        self.img_name = self.img_name + self.img_name + self.img_name

    def __getitem__(self, index):
        if self.mode == 'train':
            index_sub = index // self.sig
            index = index % self.sig
            if index_sub == 0:
                img_path = os.path.join(self.root, 'DID-MDN-training', 'Rain_Heavy', 'train2018new',
                                        self.img_name[index])
            if index_sub == 1:
                img_path = os.path.join(self.root, 'DID-MDN-training', 'Rain_Light', 'train2018new',
                                        self.img_name[index])
            if index_sub == 2:
                img_path = os.path.join(self.root, 'DID-MDN-training', 'Rain_Medium', 'train2018new',
                                        self.img_name[index])
        else:
            img_path = os.path.join(self.root, 'DID-MDN-test', self.img_name[index])
        img = Image.open(img_path)
        w, h = img.size
        # Convert grayscale images to rgb
        if img.mode != "RGB":
            img = to_rgb(img)

        clear_img = img.crop((0, 0, w / 2, h))
        rain_img = img.crop((w / 2, 0, w, h))

        if self.mode == 'train':
            item_rain, item_clear = train_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        else:
            item_rain, item_clear = val_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        return {"rain": item_rain, "clear": item_clear}

    def __len__(self):
        return len(self.img_name)


class rain200_ImageDataset(Dataset):
    def __init__(self, opt, root, transforms=None, mode="train", name='Rain200H'):
        self.transform = transforms
        self.size = opt.img_height
        self.mode = mode
        self.root = os.path.join(root, name)
        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
        else:
            self.root = os.path.join(self.root, 'test')
        self.rain_path = [os.path.join(self.root, 'rain', 'norain-' + str(i) + 'x2.png')
                          for i in range(1, len(os.listdir(os.path.join(self.root, 'rain'))) + 1)]
        self.clear_path = [os.path.join(self.root, 'norain', 'norain-' + str(i) + '.png')
                           for i in range(1, len(os.listdir(os.path.join(self.root, 'rain'))) + 1)]

    def __getitem__(self, index):
        rain_img = Image.open(self.rain_path[index])
        clear_img = Image.open(self.clear_path[index])
        # Convert grayscale images to rgb
        if rain_img.mode != "RGB":
            rain_img = to_rgb(rain_img)
        if clear_img.mode != "RGB":
            clear_img = to_rgb(clear_img)

        if self.mode == 'train':
            item_rain, item_clear = train_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        else:
            item_rain, item_clear = val_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        return {"rain": item_rain, "clear": item_clear}

    def __len__(self):
        return len(self.rain_path)

class rain1400_ImageDataset(Dataset):
    def __init__(self, opt, root, transforms=None, mode="train", name='Rain1400'):
        self.transform = transforms
        self.size = opt.img_height
        self.mode = mode
        self.root = os.path.join(root, name)
        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
        else:
            self.root = os.path.join(self.root, 'test')
        self.rain_path = [os.path.join(self.root, 'rain', 'norain-' + str(i) + '.png')
                          for i in range(1, len(os.listdir(os.path.join(self.root, 'rain'))) + 1)]
        self.clear_path = [os.path.join(self.root, 'norain', 'norain-' + str(i) + '.png')
                           for i in range(1, len(os.listdir(os.path.join(self.root, 'rain'))) + 1)]

    def __getitem__(self, index):
        rain_img = Image.open(self.rain_path[index])
        clear_img = Image.open(self.clear_path[index])
        # Convert grayscale images to rgb
        if rain_img.mode != "RGB":
            rain_img = to_rgb(rain_img)
        if clear_img.mode != "RGB":
            clear_img = to_rgb(clear_img)

        if self.mode == 'train':
            item_rain, item_clear = train_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        else:
            item_rain, item_clear = val_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        return {"rain": item_rain, "clear": item_clear}

    def __len__(self):
        return len(self.rain_path)

class rain100_ImageDataset(Dataset):
    def __init__(self, opt, root, transforms=None, mode="train", name='Rain100H'):
        self.transform = transforms
        self.size = opt.img_height
        self.mode = mode
        self.root = os.path.join(root, name)
        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
        else:
            self.root = os.path.join(self.root, 'test')
        self.rain_path = [os.path.join(self.root, 'rain', 'norain-' + str(i) + '.png')
                          for i in range(1, len(os.listdir(os.path.join(self.root, 'rain'))) + 1)]
        self.clear_path = [os.path.join(self.root, 'norain', 'norain-' + str(i) + '.png')
                           for i in range(1, len(os.listdir(os.path.join(self.root, 'rain'))) + 1)]

    def __getitem__(self, index):
        rain_img = Image.open(self.rain_path[index])
        clear_img = Image.open(self.clear_path[index])
        # Convert grayscale images to rgb
        if rain_img.mode != "RGB":
            rain_img = to_rgb(rain_img)
        if clear_img.mode != "RGB":
            clear_img = to_rgb(clear_img)

        if self.mode == 'train':
            item_rain, item_clear = train_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        else:
            item_rain, item_clear = val_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        return {"rain": item_rain, "clear": item_clear}

    def __len__(self):
        return len(self.rain_path)


class rain13k_ImageDataset(Dataset):
    def __init__(self, opt, root, transforms=None, mode="train", test_name=None):
        self.transform = transforms
        self.size = opt.img_height
        self.root = os.path.join(root, 'rain13k')
        self.mode = mode
        if self.mode == 'train':
            with open(os.path.join(self.root, mode + '.txt'), 'r') as f:
                self.rain_path = [os.path.join(self.root, 'train', 'rain', line.strip())
                                  for line in f.readlines() if len(line.strip()) > 0]
            with open(os.path.join(self.root, mode + '.txt'), 'r') as f:
                self.clear_path = [os.path.join(self.root, 'train', 'clear', line.strip())
                                   for line in f.readlines() if len(line.strip()) > 0]
        else:
            with open(os.path.join(self.root, mode + '_' + test_name + '.txt'), 'r') as f:
                self.rain_path = [os.path.join(self.root, 'test', test_name, 'rain', line.strip())
                                  for line in f.readlines() if len(line.strip()) > 0]
            with open(os.path.join(self.root, mode + '_' + test_name + '.txt'), 'r') as f:
                self.clear_path = [os.path.join(self.root, 'test', test_name, 'clear', line.strip())
                                   for line in f.readlines() if len(line.strip()) > 0]

    def __getitem__(self, index):
        rain_img = Image.open(self.rain_path[index])
        clear_img = Image.open(self.clear_path[index])

        # Convert grayscale images to rgb
        if rain_img.mode != "RGB":
            rain_img = to_rgb(rain_img)
        if clear_img.mode != "RGB":
            clear_img = to_rgb(clear_img)

        if self.mode == 'train':
            item_rain, item_clear = train_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        else:
            item_rain, item_clear = val_tran(inp_img=rain_img, tar_img=clear_img, ps=self.size)
        return {"rain": item_rain, "clear": item_clear}

    def __len__(self):
        return len(self.rain_path)


def train_tran(inp_img, tar_img, ps):
    item_rain = FF.to_tensor(inp_img)
    item_clear = FF.to_tensor(tar_img)

    aug = random.randint(0, 8)

    c, h, w = item_rain.shape
    padw = ps - w if w < ps else 0
    padh = ps - h if h < ps else 0

    # Reflect Pad in case image is smaller than patch_size
    if padw != 0 or padh != 0:
        item_rain = FF.pad(item_rain, [0, 0, padw, padh], padding_mode='reflect')
        item_clear = FF.pad(item_clear, [0, 0, padw, padh], padding_mode='reflect')

    i, j, h, w = transforms.RandomCrop.get_params(item_rain, output_size=(ps, ps))
    inp_img = FF.crop(item_rain, i, j, h, w)
    tar_img = FF.crop(item_clear, i, j, h, w)

    # Data Augmentations
    if aug == 1:
        inp_img = inp_img.flip(1)
        tar_img = tar_img.flip(1)
    elif aug == 2:
        inp_img = inp_img.flip(2)
        tar_img = tar_img.flip(2)
    elif aug == 3:
        inp_img = torch.rot90(inp_img, dims=(1, 2))
        tar_img = torch.rot90(tar_img, dims=(1, 2))
    elif aug == 4:
        inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
        tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
    elif aug == 5:
        inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
        tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
    elif aug == 6:
        inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
        tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
    elif aug == 7:
        inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
        tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
    return inp_img, tar_img


def val_tran(inp_img, tar_img, ps):
    inp_img = FF.to_tensor(inp_img)
    tar_img = FF.to_tensor(tar_img)

    return inp_img, tar_img
