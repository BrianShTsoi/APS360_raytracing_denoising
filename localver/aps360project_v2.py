# -*- coding: utf-8 -*-

import os
import random
import psutil
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import torchvision
import torchvision.io as torchvio
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Set True if running locally
LOCAL = True
# Set true to use GPU
TRY_USE_GPU = True
# Set true to attempt to save memory
TRY_SAVE_MEM = False

if not LOCAL:
    import google.colab.drive as drive
    drive.mount('/content/drive')


# Base address for dataset
BasePath = "" if LOCAL else "/content/drive/MyDrive/"
TruthPaths = [BasePath + "data/baseline_data/animations/anim1/4096/",
              #BasePath + "data/baseline_data/animations/anim2/4096/",
              BasePath + "data/baseline_data/animations/anim3/4096/"]
NoisyPaths = [BasePath + "data/baseline_data/animations/anim1/1/",
              #BasePath + "data/baseline_data/animations/anim2/1/",
              BasePath + "data/baseline_data/animations/anim3/1/"]

CudaEnabled = TRY_USE_GPU and torch.cuda.is_available()
Device = torch.device("cuda:0" if CudaEnabled else "cpu")

TrainSplit = 0.7
ValidSplit = 0.15
TestSplit = 0.15
BatchSize = 4
LearningRate = 1e-3

if TRY_USE_GPU and not CudaEnabled:
    print("Cannot train on GPU, CUDA is not available")
print("Training on " + "gpu" if CudaEnabled else "cpu")


# Windows doesn't like file names with more than 2 periods
# def safe_fname(fpath, ext=None):
#     name = fpath.replace(".", "#")
#     return name if ext is None else name + ext

def get_model_name(name, batch_size, learning_rate, epoch=None):
    """ Generate a name for the model consisting of all the hyperparameter values"""
    if epoch is None:
        return "model_{0}_bs{1}_lr{2}".format(name, batch_size, learning_rate)
    else:
        return "model_{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size, learning_rate, epoch)


# Save the current model (checkpoint) to a file
# model_path = get_model_name(xxx.name, batch_size, learning_rate, epoch)
# torch.save(xxx.state_dict(), model_path)
# --------------------------------------------------------------------------------------

class AutoencoderHalf(nn.Module):
    # 32 is what the paper starts with
    def __init__(self, start_outch=32, depth_ratio=16 / 9):
        super(AutoencoderHalf, self).__init__()

        self.start_outch = start_outch
        self.depth_ratio = depth_ratio
        self.name = "AEhalfmoredata3_sod{0}_odr{1:.2f}".format(start_outch, depth_ratio)

        start_out_ch2 = int(start_outch * depth_ratio)
        start_out_ch3 = int(start_out_ch2 * depth_ratio)

        # half the # of convolutional blocks from the paper
        self.model = nn.Sequential(
            # encoder
            nn.Conv2d(in_channels=3, out_channels=start_outch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(start_outch, start_out_ch2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(start_out_ch2, start_out_ch3, 3, 1, 1),

            # decoder
            nn.ConvTranspose2d(start_out_ch3, start_out_ch2, 3, 1, 1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.ConvTranspose2d(start_out_ch2, start_outch, 3, 1, 1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.ConvTranspose2d(start_outch, 3, 3, 1, 1)
        ).to(Device)

    def forward(self, x):
        return self.model(x)


def plt_img_tensor(tensor):
    t = torch.transpose(torch.transpose(tensor, 0, 2), 0, 1)
    plt.imshow(t.detach().cpu())
    plt.show()


def save_img_tensor(tensor, save_dir, img_name):
    t = torch.clamp(tensor * 255., min=0., max=255.).byte().detach().cpu()
    torchvio.write_png(t, save_dir + img_name + ".png", compression_level=0)


def train(model, model_dir, result_dir, train_ds, valid_ds, num_epochs=5):
    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate, weight_decay=1e-5)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BatchSize, shuffle=True,
                                               num_workers=0 if CudaEnabled else 2)

    # outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            noisy, truth = data
            recon = model(noisy)
            loss = criterion(recon, truth)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))

        for i in range(BatchSize):
            save_img_tensor(truth[i], result_dir, "truth" + str(epoch + 1) + "_" + str(i))
            save_img_tensor(recon[i], result_dir, "recon" + str(epoch + 1) + "_" + str(i))
            save_img_tensor(noisy[i], result_dir, "noisy" + str(epoch + 1) + "_" + str(i))

        if not LOCAL:
            plt_img_tensor(truth[0])
            plt_img_tensor(recon[0])

    # Save the current model (checkpoint) to a file
    model_path = model_dir + get_model_name(model.name, BatchSize, LearningRate, epoch + 1)
    torch.save(model.state_dict(), model_path)

        # outputs.append((epoch, img, recon),)
    # return outputs


def get_io_paths(noisy_dirs, truth_dirs):
    """
    noisy_dirs: List of noisy image directories.
    truth_dirs: List of truth image directories.
    """
    assert len(noisy_dirs) == len(truth_dirs)

    paths = []
    for i in range(len(noisy_dirs)):
        nfiles = [noisy_dirs[i] + f for f in os.listdir(noisy_dirs[i])]
        tfiles = [truth_dirs[i] + f for f in os.listdir(truth_dirs[i])]
        nfiles.sort()
        tfiles.sort()
        assert len(nfiles) == len(tfiles)
        paths.extend(zip(nfiles, tfiles))

    return paths


# this is the only thing that works to avoid running out of RAM in Colab
class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, io_paths, transform=None):
        """
        io_paths: list of tuples of input-output image paths.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.paths = io_paths
        self.transform = transform
        self.noisy_imgs = [None] * len(self.paths)
        self.truth_imgs = [None] * len(self.paths)

    def __len__(self):
        return len(self.paths) * 64

    def read_img(self, idx, pidx):
        return torchvio.read_image(self.paths[idx][pidx], torchvio.ImageReadMode.RGB).to(Device).to(torch.float)

    def __getitem__(self, idx):
        img_idx = int(idx / 64)
        crop_idx = idx % 64
        crop_idx_x = crop_idx % 8
        crop_idx_y = int(crop_idx / 8)

        if self.noisy_imgs[img_idx] is not None:
            noisy_img = self.noisy_imgs[img_idx] # cached
            truth_img = self.truth_imgs[img_idx]
        else:
            noisy_img = self.read_img(img_idx, 0)
            truth_img = self.read_img(img_idx, 1)
            noisy_img /= 255.
            truth_img /= 255.

            # cache tensor if under 75% mem usage
            if not TRY_SAVE_MEM or psutil.virtual_memory()[2] < 75.0:
                self.noisy_imgs[img_idx] = noisy_img
                self.truth_imgs[img_idx] = truth_img

            assert noisy_img.size() == (3, 1024, 1024)
            assert truth_img.size() == (3, 1024, 1024)

        noisy_sample = TF.crop(noisy_img, 128 * crop_idx_y, 128 * crop_idx_x, 128, 128)
        truth_sample = TF.crop(truth_img, 128 * crop_idx_y, 128 * crop_idx_x, 128, 128)

        # noisy_img = torchvio.read_image(self.paths[idx][0], torchvio.ImageReadMode.RGB).to(Device).to(torch.float)
        # truth_img = torchvio.read_image(self.paths[idx][1], torchvio.ImageReadMode.RGB).to(Device).to(torch.float)
        # noisy_img /= 255.
        # truth_img /= 255.

        # noisy_img = torchvio.read_image(self.paths[idx][0], torchvio.ImageReadMode.RGB).to(Device)
        # truth_img = torchvio.read_image(self.paths[idx][1], torchvio.ImageReadMode.RGB).to(Device).to(torch.float)
        # noisy_img = torch.cat((noisy_img, transforms.functional.rgb_to_grayscale(noisy_img))).to(Device).to(torch.float)
        # #truth_img = torch.cat((truth_img, transforms.functional.rgb_to_grayscale(truth_img))).to(torch.float).to(Device)
        # noisy_img /= 255.
        # truth_img /= 255.

        if self.transform:
            noisy_sample = self.transform(noisy_sample)
            truth_sample = self.transform(truth_sample)

        return [noisy_sample, truth_sample]


model = AutoencoderHalf(depth_ratio=16 / 3).to(Device)
model_dir = BasePath + "data/nnmodel/{}/".format(model.name)
result_dir = model_dir + get_model_name(model.name, BatchSize, LearningRate) + "_results/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

paths = get_io_paths(NoisyPaths, TruthPaths)
random.shuffle(paths)
train_eidx = int(len(paths) * TrainSplit)
valid_eidx = int(len(paths) * (TrainSplit + ValidSplit))

#trans128 = transforms.CenterCrop(size=128).to(Device)
train_ds = ImagesDataset(paths[:train_eidx]) #, transform=trans128)
valid_ds = ImagesDataset(paths[train_eidx:valid_eidx]) #, transform=trans128)
test_ds = ImagesDataset(paths[valid_eidx:]) #, transform=trans128)

fpaths = open(model_dir + "paths.txt", "w")
fpaths.write("split={0},{1},{2}".format(TrainSplit, ValidSplit, TestSplit) + "\n")
for p in paths:
    fpaths.write(str(p) + "\n")
fpaths.close()

train(model, model_dir, result_dir, train_ds, valid_ds, num_epochs=200)

# train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
# plt.imshow(np.transpose(np.array(next(iter(train_loader))[0][0]), (1, 2, 0)))

# print("Final outputs")
# plt.imshow(np.transpose(final_truth.detach().numpy()[0], (1, 2, 0)))
# plt.show()
# plt.imshow(np.transpose(final_recon.detach().numpy()[0], (1, 2, 0)))
# plt.show()
