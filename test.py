import os
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")
import torch
num_workers = 8
torch.set_num_threads(num_workers)
from glob import glob
import pandas as pd
import random
import numpy as np
from torch import nn
from model.HASSeg import HASSeg
from utils.metrics import metrics
from monai.data import list_data_collate, DataLoader
from datetime import datetime
import monai
from monai.transforms import (
    CropForegroundd,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    ToTensord,
    EnsureChannelFirstd,
    RandRotated,
)
from scipy import ndimage
torch.cuda.empty_cache()

def test(pretrained_model_path=None, use_pretrained_model=False):
    patch_size = (128, 128, 64)
    batch_size = 2 # Note: Each case takes two images —— num_images = batch_size * 2
    num_channels = [8, 16, 32, 64, 128]
    window_size = [8, 8, 4]

    # set file path
    date = datetime.now().strftime("%Y_%m_%d %H-%M")
    ct_path = "autopet_fdg_tumor/imagesTr"
    pet_path = "autopet_fdg_tumor/imagesTr"
    label_path = "autopet_fdg_tumor/labelsTr"
    images_CT = sorted(glob(os.path.join(ct_path, "*0000.nii.gz")))
    images_PET = sorted(glob(os.path.join(pet_path, "*0001.nii.gz")))
    labels = sorted(glob(os.path.join(label_path, "*.nii.gz")))

    # set save path
    save_path = f"./save/autopet2024_HADSeg_test/{date}/"
    os.makedirs(save_path, exist_ok=True)

    # dataset split
    split = 0.6
    split_1 = 0.8

    # setting of model's input
    in_channels = 1
    n_classes = 2

    # setting of epoch
    epochs = 200



    # fixing random seed
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # data transform
    test_transforms = Compose(
            [
                LoadImaged(keys=["img", "img_ct", "seg"]),
                EnsureChannelFirstd(keys=["img", "img_ct","seg"], channel_dim="no_channel"),
                CropForegroundd(keys=["img","img_ct","seg"], source_key="img", select_fn=lambda x:x>x.min()),

                RandCropByPosNegLabeld(
                    keys=["img","img_ct","seg"],
                    label_key="seg",
                    spatial_size=patch_size,
                    pos=1,
                    neg=1,
                    num_samples=2,
                ),
                ToTensord(keys=["img", "img_ct"]),
            ]
        )

    length = len(images_CT)
    print("The number of samples:", length)

    # split dataset
    test_images = images_PET[int(split_1 * length):]
    test_ct_images = images_CT[int(split_1 * length):]
    test_labels = labels[int(split_1 * length):]

    test_files = [{"img": img, "img_ct": ct,"seg": seg} \
                    for img, ct, seg in zip(test_images, test_ct_images, test_labels)]
    print("Test set includes:", len(test_images))

    # create dataloader
    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        )

    # create model and load pretrained weight
    model = HASSeg(
                patch_size=patch_size,
                in_channels=in_channels,
                n_classes=n_classes,
                num_channels=num_channels,
                window_size=window_size).to(device)
    if use_pretrained_model:
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint,strict=False)
        print(f'load checkpoint from {pretrained_model_path}\n')
    # 使用 DataParallel 自动分配到多个 GPU 上
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model,device_ids=[0,1])

    print("\n"*2, "*" * 20, "|| Test ||", "*" * 20)
    model.eval()
    with torch.no_grad():
        for i,test_data in enumerate(test_loader):
            test_ct, test_pet, test_labels = test_data["img_ct"].type(torch.FloatTensor).to(device),\
                            test_data["img"].type(torch.FloatTensor).to(device),\
                            test_data["seg"].type(torch.LongTensor).to(device)
            r = model(test_ct, test_pet)[0]
            test_labels[test_labels!=0] = 1
            # compute metric for current iteration
            r = r.argmax(dim=1, keepdim=True)
            fp, fn, iou, dice, precision, recall, hd95 = metrics(test_labels, r)
            print(f"iter {i + 1}/{len(test_loader)}")
            print("Test {} [FP:{:.4f}, FN:{:.4f}, IoU:{:.4f}, Dice:{:.4f} Precision:{:.4f} Recall:{:.4f} HD95:{:.4f} pix:{:6}/{:6}]"\
                    .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), fp, fn, iou, dice, precision, recall, hd95, r.sum(), test_labels.sum()))
            metric_list += [[fp, fn, iou, dice, precision, recall, hd95]]

        mean_metric = np.mean(metric_list, axis=0)


        print(f"Test Outcome: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("[FP:{:.4f}, FN:{:.4f}, IoU:{:.4f}, Dice:{:.4f}, Precision:{:.4f} Recall:{:.4f} HD95:{:.4f}]"
                .format(mean_metric[0],
                        mean_metric[1],
                        mean_metric[2],
                        mean_metric[3],
                        mean_metric[4],
                        mean_metric[5],
                        mean_metric[6]))
        print("*" * 20, "|| Final test ||", "*" * 20, "\n"*2)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    pretrained_model_path = "save/autopet2024_HADSeg/2025_04_26 08-01/val_best.pth"
    use_pretrained_model = True
    test(pretrained_model_path, use_pretrained_model)