import time
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import nibabel as nib
import torch
import numpy as np
from model.H2ASeg import H2ASeg
from monai.data import list_data_collate, DataLoader
import monai
from monai.transforms import (
    CropForegroundd,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    ToTensord,
    AddChanneld,
    RandRotated,
    EnsureChannelFirstd
)
from sklearn.model_selection import KFold

def main():
    patch_size = (128, 128, 64)
    batch_size = 2 # Note: Each case takes two images —— num_images = batch_size * 2
    num_channels = [8, 16, 32, 64, 128]
    window_size = [8, 8, 4]
    pretrained_model_path = "save/autopet2024_HADSeg/2025_04_20 23-18/val_best.pth"
    use_pretrained_model = True
    loss_weights = [5, 4, 3, 2, 1]

    # set file path
    date = datetime.now().strftime("%Y_%m_%d %H-%M")
    ct_path = "autopet_fdg_tumor/imagesTr"
    pet_path = "autopet_fdg_tumor/imagesTr"
    label_path = "autopet_fdg_tumor/labelsTr"
    images_CT = sorted(glob(os.path.join(ct_path, "*0000.nii.gz")))
    images_PET = sorted(glob(os.path.join(pet_path, "*0001.nii.gz")))
    labels = sorted(glob(os.path.join(label_path, "*.nii.gz")))

    # set save path
    save_path = f"./save/autopet2024_HADSeg/{date}/"
    os.makedirs(save_path, exist_ok=True)

    # dataset split
    split = 0.8
    split_1 = 1

    # setting of model's input
    in_channels = 1
    n_classes = 2

    # setting of epoch
    epochs = 200
    save_model_interval = 10
    val_interval = 10

    # adjustment of learning rate
    lr = 1e-5
    step_size = 1
    gamma = 0.99

    # fixing random seed
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # print metric
    deep = True

    # dice & bceloss
    criterion_dice = monai.losses.DiceLoss(include_background=True,
                                           to_onehot_y=True,
                                           softmax=True).to(device)
    criterion_ce = nn.CrossEntropyLoss().to(device)

    # data transform
    train_transforms = Compose(
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
                RandRotated(keys=['img', 'img_ct', 'seg'], prob=0.8, range_z=45),
                ToTensord(keys=["img", "img_ct"]),
            ]
        )

    val_transforms = Compose(
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
    train_images = images_PET[: int(split * length)]
    train_ct_images = images_CT[: int(split * length)]
    train_labels = labels[: int(split * length)]

    test_images = images_PET[int(split * length):int(split_1*length)]
    test_ct_images = images_CT[int(split * length):int(split_1*length)]
    test_labels = labels[int(split * length):int(split_1*length)]

    train_files = [{"img": img, "img_ct": ct,"seg": seg} \
                    for img, ct, seg in zip(train_images, train_ct_images, train_labels)]
    val_files = [{"img": img, "img_ct": ct,"seg": seg} \
                    for img, ct, seg in zip(test_images, test_ct_images, test_labels)]
    print("Training set includes:", len(train_images))
    print("Validation set includes:", len(test_images))

    # create dataloader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        )

    # create model and load pretrained weight
    model = H2ASeg(
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

    # model = model.to(device=0)

    # Ready for training
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=lr / 10)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    epoch_loss_metric_list = []
    writer = SummaryWriter(save_path + 'logs')

    # training start
    print("*" * 20, "|| Training ||", "*" * 20)
    # iteration = 0
    best_dice = 0
    best_train_dice = 0
    for epoch in range(epochs):
        start = time.time()
        model.train()
        loss_metric_list = []

        for step, batch_data in enumerate(train_loader):
            # iteration += 1
            ct, pet, labels = batch_data["img_ct"].type(torch.FloatTensor).to(device),\
                              batch_data["img"].type(torch.FloatTensor).to(device),\
                              batch_data["seg"].type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            outputs = model(ct, pet)
            labels[labels!=0] = 1

            # calculate loss
            loss1 = criterion_ce(outputs[0], labels.squeeze(1)) + criterion_dice(input=outputs[0], target=labels)
            loss2 = criterion_ce(outputs[1], labels.squeeze(1)) + criterion_dice(input=outputs[1], target=labels)
            loss3 = criterion_ce(outputs[2], labels.squeeze(1)) + criterion_dice(input=outputs[2], target=labels)
            loss4 = criterion_ce(outputs[3], labels.squeeze(1)) + criterion_dice(input=outputs[3], target=labels)
            loss5 = criterion_ce(outputs[4], labels.squeeze(1)) + criterion_dice(input=outputs[4], target=labels)
            loss = loss1 * loss_weights[0] + loss2 * loss_weights[1] + loss3 * loss_weights[2] + \
                        loss4 * loss_weights[3] + loss5 * loss_weights[4]
            loss.backward()
            optimizer.step()
            l = loss.item()
            l1 = loss1.item()
            l2 = loss2.item()
            l3 = loss3.item()
            l4 = loss4.item()
            l5 = loss5.item()

            epoch_len = len(train_ds) // batch_size
            # print metric
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {epoch+1}/{epochs} {step}/{epoch_len} ")
            print(f"[Training Loss:{l:.4f}, Loss1:{l1:.4f}, Loss2:{l2:.4f}, Loss3:{l3:.4f}, Loss4:{l4:.4f}, Loss5:{l5:.4f}]")

            fp, fn, iou, dice = show_deep_metrics(outputs, labels, deep)

            # writer.add_scalar("Training Loss", l, iteration)
            # writer.add_scalar("Training Loss1", l1, iteration)
            # writer.add_scalar("Training Loss2", l2, iteration)
            # writer.add_scalar("Training Loss3", l3, iteration)
            # writer.add_scalar("Training Loss4", l4, iteration)
            # writer.add_scalar("Training Loss5", l5, iteration)
            # writer.add_scalar('Training FP', fp, iteration)
            # writer.add_scalar('Training FN', fn, iteration)
            # writer.add_scalar('Training Dice', dice, iteration)
            loss_metric_list += [[l, fp, fn, iou, dice]]
        scheduler.step()

        # save average value of [loss, fp, fn, iou, dice] in every epoch
        epoch_loss_metric_list += [np.mean(loss_metric_list, axis=0).tolist()]

        if (epoch+1) % save_model_interval == 0:
            torch.save(model.state_dict(), save_path + f"{epoch}.pth")

        now_dice = epoch_loss_metric_list[-1][-1]
        if now_dice >= best_train_dice:
            print(f"get new best dice {best_train_dice} -> {now_dice}, save new 'train_best.pth'")
            best_train_dice = now_dice
            torch.save(model.state_dict(), save_path + "train_best.pth")

        print(f"training epoch {epoch + 1}: best training_epoch_dice == {best_train_dice}")
        print(f"training epoch {epoch + 1}: average [loss:{epoch_loss_metric_list[-1][0]}, ", end="")
        print(f"fp:{epoch_loss_metric_list[-1][1]}, fn:{epoch_loss_metric_list[-1][2]}, ", end="")
        print(f"iou:{epoch_loss_metric_list[-1][3]}, dice:{epoch_loss_metric_list[-1][4]}]")
        print(f"training epoch {epoch + 1}: time cost {time.time() - start} s")
        writer.add_scalar("Training Loss", epoch_loss_metric_list[-1][0], epoch + 1)
        writer.add_scalar('Training FP', epoch_loss_metric_list[-1][1], epoch + 1)
        writer.add_scalar('Training FN', epoch_loss_metric_list[-1][2], epoch + 1)
        writer.add_scalar("Training IoU", epoch_loss_metric_list[-1][3], epoch + 1)
        writer.add_scalar('Training Dice', epoch_loss_metric_list[-1][4], epoch + 1)

        torch.cuda.empty_cache()
        # validation
        if (epoch + 1) % val_interval == 0:
            print("\n"*2, "*" * 20, "|| Validating ||", "*" * 20)
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                metric_list = []
                for val_data in val_loader:
                    val_ct, val_pet, val_labels = val_data["img_ct"].type(torch.FloatTensor).to(device),\
                                    val_data["img"].type(torch.FloatTensor).to(device),\
                                    val_data["seg"].type(torch.LongTensor).to(device)
                    r = model(val_ct, val_pet)[0]
                    val_labels[val_labels!=0] = 1
                    # compute metric for current iteration
                    r = r.argmax(dim=1, keepdim=True)
                    fp, fn, iou, dice = metrics_tensor(val_labels, r)
                    print("val {} [FP:{:.4f}, FN:{:.4f}, IoU:{:.4f}, Dice:{:.4f} pix:{:6}/{:6}]"\
                            .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), fp, fn, iou, dice, r.sum(), val_labels.sum()))

                    metric_list += [[fp, fn, iou, dice]]

            mean_metric = np.mean(metric_list, axis=0)

            now_dice = mean_metric[3]
            if now_dice >= best_dice:
                print(f"get new best dice {best_dice} -> {now_dice}, save new 'val_best.pth'")
                best_dice = now_dice
                torch.save(model.state_dict(), save_path + "val_best.pth")

            print(f"epoch {epoch + 1}/{epochs}")
            print(f"epoch {epoch + 1}: best validating_epoch_dice == {best_dice}")
            print("[FP:{:.4f}, FN:{:.4f}, IoU:{:.4f}, Dice:{:.4f}]".format(mean_metric[0],
                                                                           mean_metric[1],
                                                                           mean_metric[2],
                                                                           mean_metric[3]))
            writer.add_scalar("Validating FP", mean_metric[0], epoch + 1)
            writer.add_scalar('Validating FN', mean_metric[1], epoch + 1)
            writer.add_scalar('Validating IoU', mean_metric[2], epoch + 1)
            writer.add_scalar('Validating Dice', mean_metric[3], epoch + 1)

            print("*" * 20, "|| Final Validating ||", "*" * 20, "\n"*2)
            torch.cuda.empty_cache()
    writer.close()
    
def segmentation_visualization(output, ct, pet, labels, save_dir):
    save_dir = save_dir + f"visualization"
    os.makedirs(save_dir, exist_ok=True)


    batch_size = int(output.shape[0] / 2)


    output = output.view(batch_size*2, 2, *output.shape[2:]).cpu()
    output = output.argmax(dim=1)


    pet_image_batch = pet[:batch_size].cpu().numpy()
    ct_image_batch = ct[:batch_size].cpu().numpy()
    gt_segmentation_batch= labels[:batch_size].cpu().numpy()


    # pet_image_batch = pet_image_batch.squeeze(0)
    # ct_image_batch = ct_image_batch.squeeze(0)
    # gt_segmentation_batch = gt_segmentation_batch.squeeze(0)


    fig, axes = plt.subplots(batch_size, 4, figsize=(20, 5 * batch_size), dpi=300)
    axes = axes.ravel()


    for idx in range (batch_size):
        pet_image = pet_image_batch[idx]
        ct_image = ct_image_batch[idx]
        gt_segmentation = gt_segmentation_batch[idx]
        mask = output[idx]

        pet_image = pet_image.squeeze(0)
        ct_image = ct_image.squeeze(0)
        gt_segmentation = gt_segmentation.squeeze(0)

        slice_idx = np.argmax(np.sum(gt_segmentation, axis=(0, 1)))


        pet_slice = pet_image[:, :, slice_idx]
        ct_slice = ct_image[:, :, slice_idx]
        gt_slice = gt_segmentation[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]



        true_positive = np.logical_and(mask_slice == 1, gt_slice == 1)
        false_negative = np.logical_and(mask_slice == 0, gt_slice == 1)
        false_positive = np.logical_and(mask_slice == 1, gt_slice == 0)



        axes[0 + idx * 4].imshow(pet_slice, cmap='gray')
        axes[0 + idx * 4].set_title(f'PET', fontsize=32)
        axes[0 + idx * 4].axis('off')

        axes[1 + idx * 4].imshow(ct_slice, cmap='gray')
        axes[1 + idx * 4].set_title(f'CT', fontsize=32)
        axes[1 + idx * 4].axis('off')

        axes[2 + idx * 4].imshow(np.where(gt_slice == 0, 0, 1), cmap='gray')
        axes[2 + idx * 4].set_title(f'GT', fontsize=32)
        axes[2 + idx * 4].axis('off')

        axes[3 + idx * 4].imshow(ct_slice, cmap='gray')

        red_cmap = mcolors.ListedColormap(['red'])
        blue_cmap = mcolors.ListedColormap(['blue'])
        yellow_cmap = mcolors.ListedColormap(['yellow'])
        ct_slice[:, :] = 255
        axes[3 + idx * 4].imshow(np.ma.masked_where(true_positive == False, ct_slice),
                                 alpha=0.7, cmap=red_cmap)
        axes[3 + idx * 4].imshow(np.ma.masked_where(false_negative == False, ct_slice),
                                 alpha=0.7, cmap=blue_cmap)
        axes[3 + idx * 4].imshow(np.ma.masked_where(false_positive == False, ct_slice),
                                 alpha=0.7, cmap=yellow_cmap)
        axes[3 + idx * 4].set_title(f'Segmentation', fontsize=32)
        axes[3 + idx * 4].axis('off')


    save_path =os.path.join(save_dir, f'sample_{step + 1}_slice_{slice_idx}.png')
    plt.savefig(save_path, dpi=3000, bbox_inches='tight')
    plt.close(fig)

def show_png(output, ct, pet, labels, save_dir, epoch, step):
    save_dir = save_dir + f"visual{epoch + 1}"
    os.makedirs(save_dir, exist_ok=True)


    batch_size = int(output.shape[0] / 2)


    output = output.view(batch_size*2, 2, *output.shape[2:]).cpu()
    output = output.argmax(dim=1)


    pet_image_batch = pet[:batch_size].cpu().numpy()
    ct_image_batch = ct[:batch_size].cpu().numpy()
    gt_segmentation_batch= labels[:batch_size].cpu().numpy()


    # pet_image_batch = pet_image_batch.squeeze(0)
    # ct_image_batch = ct_image_batch.squeeze(0)
    # gt_segmentation_batch = gt_segmentation_batch.squeeze(0)


    fig, axes = plt.subplots(batch_size, 4, figsize=(20, 5 * batch_size), dpi=300)
    axes = axes.ravel()


    for idx in range (batch_size):
        pet_image = pet_image_batch[idx]
        ct_image = ct_image_batch[idx]
        gt_segmentation = gt_segmentation_batch[idx]
        mask = output[idx]

        pet_image = pet_image.squeeze(0)
        ct_image = ct_image.squeeze(0)
        gt_segmentation = gt_segmentation.squeeze(0)

        slice_idx = np.argmax(np.sum(gt_segmentation, axis=(0, 1)))


        pet_slice = pet_image[:, :, slice_idx]
        ct_slice = ct_image[:, :, slice_idx]
        gt_slice = gt_segmentation[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]



        true_positive = np.logical_and(mask_slice == 1, gt_slice == 1)
        false_negative = np.logical_and(mask_slice == 0, gt_slice == 1)
        false_positive = np.logical_and(mask_slice == 1, gt_slice == 0)



        axes[0 + idx * 4].imshow(pet_slice, cmap='gray')
        axes[0 + idx * 4].set_title(f'PET', fontsize=32)
        axes[0 + idx * 4].axis('off')

        axes[1 + idx * 4].imshow(ct_slice, cmap='gray')
        axes[1 + idx * 4].set_title(f'CT', fontsize=32)
        axes[1 + idx * 4].axis('off')

        axes[2 + idx * 4].imshow(np.where(gt_slice == 0, 0, 1), cmap='gray')
        axes[2 + idx * 4].set_title(f'GT', fontsize=32)
        axes[2 + idx * 4].axis('off')

        axes[3 + idx * 4].imshow(ct_slice, cmap='gray')

        red_cmap = mcolors.ListedColormap(['red'])
        blue_cmap = mcolors.ListedColormap(['blue'])
        yellow_cmap = mcolors.ListedColormap(['yellow'])
        ct_slice[:, :] = 255
        axes[3 + idx * 4].imshow(np.ma.masked_where(true_positive == False, ct_slice),
                                 alpha=0.7, cmap=red_cmap)
        axes[3 + idx * 4].imshow(np.ma.masked_where(false_negative == False, ct_slice),
                                 alpha=0.7, cmap=blue_cmap)
        axes[3 + idx * 4].imshow(np.ma.masked_where(false_positive == False, ct_slice),
                                 alpha=0.7, cmap=yellow_cmap)
        axes[3 + idx * 4].set_title(f'Segmentation', fontsize=32)
        axes[3 + idx * 4].axis('off')


    save_path =os.path.join(save_dir, f'sample_{step + 1}_slice_{slice_idx}.png')
    plt.savefig(save_path, dpi=3000, bbox_inches='tight')
    plt.close(fig)




def show_front_png(output, ct, pet, labels, save_dir, step):
    # print(output.shape)
    save_dir = os.path.join(save_dir, "front2")
    os.makedirs(save_dir, exist_ok=True)


    batch_size = int(output.shape[0])


    output = output.reshape(batch_size, 1, *output.shape[2:])
    # output = output.argmax(dim=1)


    pet_image_batch = pet[:batch_size].cpu().numpy()
    ct_image_batch = ct[:batch_size].cpu().numpy()
    gt_segmentation_batch= labels[:batch_size].cpu().numpy()


    # pet_image_batch = pet_image_batch.squeeze(0)
    # ct_image_batch = ct_image_batch.squeeze(0)
    # gt_segmentation_batch = gt_segmentation_batch.squeeze(0)


    fig, axes = plt.subplots(batch_size, 4, figsize=(20, 5 * batch_size), dpi=300)
    axes = axes.ravel()


    for idx in range (batch_size):
        pet_image = pet_image_batch[idx]
        ct_image = ct_image_batch[idx]
        gt_segmentation = gt_segmentation_batch[idx]
        mask = output[idx]

        pet_image = pet_image.squeeze(0)
        ct_image = ct_image.squeeze(0)
        gt_segmentation = gt_segmentation.squeeze(0)
        mask = mask.squeeze(0)

        # print(pet_image.shape)
        # print(ct_image.shape)
        # print(gt_segmentation.shape)
        # print(mask.shape)
        # 计算每个切片中的 True Positive 数量
        true_positive_per_slice = np.sum(
            np.logical_and(mask == 1, gt_segmentation == 1), axis=(0, 2)
        )

        # 找到 True Positive 数量最大的切片索引
        slice_idx = np.argmax(true_positive_per_slice)


        pet_slice = pet_image[:, slice_idx, :]
        ct_slice = ct_image[:, slice_idx, :]
        gt_slice = gt_segmentation[:, slice_idx, :]
        mask_slice = mask[:, slice_idx, :]



        true_positive = np.logical_and(mask_slice == 1, gt_slice == 1)
        false_negative = np.logical_and(mask_slice == 0, gt_slice == 1)
        false_positive = np.logical_and(mask_slice == 1, gt_slice == 0)


        axes[0 + idx * 4].imshow(pet_slice, cmap='gray')
        axes[0 + idx * 4].set_title(f'PET', fontsize=32)
        axes[0 + idx * 4].axis('off')

        axes[1 + idx * 4].imshow(ct_slice, cmap='gray')
        axes[1 + idx * 4].set_title(f'CT', fontsize=32)
        axes[1 + idx * 4].axis('off')

        axes[2 + idx * 4].imshow(np.where(gt_slice == 0, 0, 1), cmap='gray')
        axes[2 + idx * 4].set_title(f'GT', fontsize=32)
        axes[2 + idx * 4].axis('off')

        axes[3 + idx * 4].imshow(ct_slice, cmap='gray')

        red_cmap = mcolors.ListedColormap(['red'])
        blue_cmap = mcolors.ListedColormap(['blue'])
        yellow_cmap = mcolors.ListedColormap(['yellow'])
        ct_slice[:, :] = 255
        axes[3 + idx * 4].imshow(np.ma.masked_where(true_positive == False, ct_slice),
                                 alpha=0.7, cmap=red_cmap)
        axes[3 + idx * 4].imshow(np.ma.masked_where(false_negative == False, ct_slice),
                                 alpha=0.7, cmap=blue_cmap)
        axes[3 + idx * 4].imshow(np.ma.masked_where(false_positive == False, ct_slice),
                                 alpha=0.7, cmap=yellow_cmap)
        axes[3 + idx * 4].set_title(f'Segmentation', fontsize=32)
        axes[3 + idx * 4].axis('off')


    save_path =os.path.join(save_dir, f'sample_{step + 1}_slice_{slice_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'save sample_{step + 1}_slice_{slice_idx}.png')
    plt.close(fig)



def show_crop2nifti(data_pairs, best_models, k_folds = 8, save_dir = ""):
    # 最终评估阶段
    print("Starting final evaluation on the entire dataset")
    patch_size =  (128, 128, 64)
    num_channels = [8, 16, 32, 64, 128]
    window_size = [8, 8, 4]

    # setting of model's input
    in_channels = 1
    n_classes = 2

    # save path
    kfold = KFold(n_splits=k_folds, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_pairs)):
        print(f'Fold {fold + 1}/{k_folds}')
        start = time.time()
        test_transforms = Compose(
            [
                LoadImaged(keys=["img", "img_ct", "seg"], image_only=True),
                EnsureChannelFirstd(keys=["img", "img_ct", "seg"]),
                # CropForegroundd(keys=["img", "img_ct", "seg"], source_key="img"),
                ToTensord(keys=["img", "img_ct"]),
            ]
        )
        # 根据索引生成训练集和验证集
        # print(train_idx, val_idx)
        # train_files = [data_pairs[i] for i in train_idx]
        val_files = [data_pairs[40]]

        test_ds = monai.data.Dataset(data=val_files, transform=test_transforms)
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            num_workers=0,
            collate_fn=list_data_collate,
        )
        # create model and load pretrained weight
        model = H2ASeg(
            patch_size=patch_size,
            in_channels=in_channels,
            n_classes=n_classes,
            num_channels=num_channels,
            window_size=window_size).cuda()

        # 使用DataParallel 自动分配到多个GPU上
        # print(torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPU!")
            model = torch.nn.DataParallel(model)

        model.load_state_dict(torch.load("/home/kasm-user/H2ASeg-main/save/hecktor2022_H2ASeg/11_29/val_best.pth"))  # 加载每个折叠的最佳模型
        model.eval()


        with torch.no_grad():
            for step, test_data in enumerate(test_loader):
                final_prediction = None
                test_ct, test_pet, test_label = test_data["img_ct"].type(torch.FloatTensor).cuda(), \
                    test_data["img"].type(torch.FloatTensor).cuda(), \
                    test_data["seg"].type(torch.FloatTensor).cuda()
                # print(type(test_ct), type(test_pet), type(test_label))
                original_h, original_w, original_d = test_ct.shape[2:]
                # print(test_ct.shape[2:])

                # 获取滑动窗口裁剪
                patches_ct = sliding_window_crop(test_ct)
                patches_pet = sliding_window_crop(test_pet)
                patches_label = sliding_window_crop(test_label)
                # print("--------------------------------", patches_label[-1].shape)
                patches_ct_ds = monai.data.Dataset(data=patches_ct)
                patches_ct_loader = DataLoader(
                    patches_ct_ds,
                    batch_size=1,
                    num_workers=0,
                    collate_fn=list_data_collate,
                )
                patches_pet_ds = monai.data.Dataset(data=patches_pet)
                patches_pet_loader = DataLoader(
                    patches_pet_ds,
                    batch_size=1,
                    num_workers=0,
                    collate_fn=list_data_collate,
                )
                patches_ds = monai.data.Dataset(data=patches_label)
                patches_loader = DataLoader(
                    patches_ds,
                    batch_size=1,
                    num_workers=0,
                    collate_fn=list_data_collate,
                )
                # 预测每个补丁
                for patch_ct, patch_pet in zip(patches_ct_loader, patches_pet_loader):
                    # print(patch_ct.squeeze(0).shape)
                # for patch_label in patches_loader:
                    r = model(patch_ct.squeeze(0).cuda(), patch_pet.squeeze(0).cuda())[0]
                    r = r.argmax(dim=1, keepdim=True)
                    # patch_label = np.array(patch_label)
                    # r = patch_label.cuda()
                    # 如果是第一次，初始化 final_prediction
                    if final_prediction is None:
                        final_prediction = r.cpu().numpy()  # 转换为 NumPy 数组
                    else:
                        final_prediction = np.concatenate((final_prediction, r.cpu().numpy()), axis=0)  # 合并结果
                # 保存为 .nii.gz 文件
                # print(f"Final prediction shape: {final_prediction.shape}")
                # 假设 final_prediction 是预测结果，形状为 (24, 1, 128, 128, 64)
                combined_prediction = np.sum(final_prediction, axis=0)  # 形状为 (1, 128, 128, 64)

                # 创建掩码，值为1的区域表示预测的目标
                mask = (combined_prediction > 0).astype(np.uint8)

                # 找到非零区域的边界
                non_zero_coords = np.where(mask[0] > 0)
                if non_zero_coords[0].size == 0:
                    print("没有找到非零区域，返回全零图像。")
                    final_prediction_reshaped = np.zeros((1, 1, original_h, original_w, original_d))
                else:
                    min_z, max_z = non_zero_coords[0].min(), non_zero_coords[0].max()
                    min_y, max_y = non_zero_coords[1].min(), non_zero_coords[1].max()
                    min_x, max_x = non_zero_coords[2].min(), non_zero_coords[2].max()

                    # 裁剪图像
                    cropped_prediction = combined_prediction[0, min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]

                    # 创建最终的全零图像，并将裁剪的部分放入其中
                    final_prediction_reshaped = np.zeros((1, 1, original_h, original_w, original_d))

                    # 确保裁剪的部分放入最终的图像中
                    z_offset = (original_h - cropped_prediction.shape[0]) // 2
                    y_offset = (original_w - cropped_prediction.shape[1]) // 2
                    x_offset = (original_d - cropped_prediction.shape[2]) // 2

                    final_prediction_reshaped[0, 0, z_offset:z_offset + cropped_prediction.shape[0],
                    y_offset:y_offset + cropped_prediction.shape[1],
                    x_offset:x_offset + cropped_prediction.shape[2]] = cropped_prediction

                # final_prediction_reshaped = final_prediction.reshape((1, 1, original_h, original_w, original_d))  # 根据需要调整形状
                affine = np.eye(4)  # 根据需要设置仿射变换矩阵
                nifti_img = nib.Nifti1Image(final_prediction_reshaped, affine)
                save_file = f"{fold + 1}_prediction_sample_{step + 1}.nii.gz"
                print(f"save nifti_img \"{save_file}\"")
                print(f"Fold {fold + 1}: time cost {time.time() - start} s \n")
                nib.save(nifti_img, os.path.join(save_dir, save_file))


def show_whole_body(images_CT, images_PET, labels, best_model, k_folds = 8, save_dir = ""):
    # 最终评估阶段
    print("Starting final evaluation on the entire dataset")
    patch_size =  (128, 128, 64)
    num_channels = [8, 16, 32, 64, 128]
    window_size = [8, 8, 4]

    # setting of model's input
    in_channels = 1
    n_classes = 2

    # save path
    # kfold = KFold(n_splits=k_folds, shuffle=False)

    # for fold, (train_idx, val_idx) in enumerate(kfold.split(data_pairs)):
    # print(f'Fold {fold + 1}/{k_folds}')

    split = 0.6
    split_1 = 0.8

    length = len(images_CT)

    # split dataset
    train_images = images_PET[: int(split * length)]
    train_ct_images = images_CT[: int(split * length)]
    train_labels = labels[: int(split * length)]

    test_images = images_PET[int(split * length):int(split_1*length)]
    test_ct_images = images_CT[int(split * length):int(split_1*length)]
    test_labels = labels[int(split * length):int(split_1*length)]

    train_files = [{"img": img, "img_ct": ct,"seg": seg} \
                    for img, ct, seg in zip(train_images, train_ct_images, train_labels)]
    val_files = [{"img": img, "img_ct": ct,"seg": seg} \
                    for img, ct, seg in zip(test_images, test_ct_images, test_labels)]

    start = time.time()
    test_transforms = Compose(
        [
            LoadImaged(keys=["img", "img_ct", "seg"], image_only=True),
            EnsureChannelFirstd(keys=["img", "img_ct", "seg"]),
            # CropForegroundd(keys=["img", "img_ct", "seg"], source_key="img"),
            ToTensord(keys=["img", "img_ct"]),
        ]
    )
    # 根据索引生成训练集和验证集
    # print(train_idx, val_idx)
    # train_files = [data_pairs[i] for i in train_idx]
    # val_files = [data_pairs[i] for i in val_idx]

    test_ds = monai.data.Dataset(data=train_files, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=0,
        collate_fn=list_data_collate,
    )
    # create model and load pretrained weight
    model = H2ASeg(
        patch_size=patch_size,
        in_channels=in_channels,
        n_classes=n_classes,
        num_channels=num_channels,
        window_size=window_size).cuda()

    # # 使用DataParallel 自动分配到多个GPU上
    # # print(torch.cuda.device_count())
    # if torch.cuda.device_count() > 1:
    #     print(f"Let's use {torch.cuda.device_count()} GPU!")
    #     model = torch.nn.DataParallel(model)

    # model.load_state_dict(torch.load(best_models[fold]))  # 加载每个折叠的最佳模型
    model.load_state_dict(torch.load(best_model))
    model.eval()

    with torch.no_grad():
        for step, test_data in enumerate(test_loader):
            final_prediction = None
            test_ct, test_pet, test_label = test_data["img_ct"].type(torch.FloatTensor).cuda(), \
                test_data["img"].type(torch.FloatTensor).cuda(), \
                test_data["seg"].type(torch.FloatTensor).cuda()
            # print(type(test_ct), type(test_pet), type(test_label))
            original_h, original_w, original_d = test_ct.shape[2:]
            # print(test_ct.shape[2:])

            # 获取滑动窗口裁剪
            patches_ct = sliding_window_crop(test_ct)
            patches_pet = sliding_window_crop(test_pet)
            patches_label = sliding_window_crop(test_label)
            # print("--------------------------------", patches_label[-1].shape)
            patches_ct_ds = monai.data.Dataset(data=patches_ct)
            patches_ct_loader = DataLoader(
                patches_ct_ds,
                batch_size=1,
                num_workers=0,
                collate_fn=list_data_collate,
            )
            patches_pet_ds = monai.data.Dataset(data=patches_pet)
            patches_pet_loader = DataLoader(
                patches_pet_ds,
                batch_size=1,
                num_workers=0,
                collate_fn=list_data_collate,
            )
            patches_ds = monai.data.Dataset(data=patches_label)
            patches_loader = DataLoader(
                patches_ds,
                batch_size=1,
                num_workers=0,
                collate_fn=list_data_collate,
            )
            # 预测每个补丁
            for patch_ct, patch_pet in zip(patches_ct_loader, patches_pet_loader):
                # print(patch_ct.squeeze(0).shape)
                # for patch_label in patches_loader:
                r = model(patch_ct.squeeze(0).cuda(), patch_pet.squeeze(0).cuda())[0]
                r = r.argmax(dim=1, keepdim=True)
                # patch_label = np.array(patch_label)
                # r = patch_label.cuda()
                # 如果是第一次，初始化 final_prediction
                if final_prediction is None:
                    final_prediction = r.cpu().numpy()  # 转换为 NumPy 数组
                else:
                    final_prediction = np.concatenate((final_prediction, r.cpu().numpy()), axis=0)  # 合并结果
            # 保存为 .nii.gz 文件
            # print(f"Final prediction shape: {final_prediction.shape}")
            # 假设 final_prediction 是预测结果，形状为 (24, 1, 128, 128, 64)
            combined_prediction = np.sum(final_prediction, axis=0)  # 形状为 (1, 128, 128, 64)

            # 创建掩码，值为1的区域表示预测的目标
            mask = (combined_prediction > 0).astype(np.uint8)

            # 找到非零区域的边界
            non_zero_coords = np.where(mask[0] > 0)
            if non_zero_coords[0].size == 0:
                print("没有找到非零区域，返回全零图像。")
                final_prediction_reshaped = np.zeros((1, 1, original_h, original_w, original_d))
            else:
                min_z, max_z = non_zero_coords[0].min(), non_zero_coords[0].max()
                min_y, max_y = non_zero_coords[1].min(), non_zero_coords[1].max()
                min_x, max_x = non_zero_coords[2].min(), non_zero_coords[2].max()

                # 裁剪图像
                cropped_prediction = combined_prediction[0, min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]

                # 创建最终的全零图像，并将裁剪的部分放入其中
                final_prediction_reshaped = np.zeros((1, 1, original_h, original_w, original_d))

                # 确保裁剪的部分放入最终的图像中
                z_offset = (original_h - cropped_prediction.shape[0]) // 2
                y_offset = (original_w - cropped_prediction.shape[1]) // 2
                x_offset = (original_d - cropped_prediction.shape[2]) // 2

                final_prediction_reshaped[0, 0, z_offset:z_offset + cropped_prediction.shape[0],
                y_offset:y_offset + cropped_prediction.shape[1],
                x_offset:x_offset + cropped_prediction.shape[2]] = cropped_prediction

            # final_prediction_reshaped = final_prediction.reshape((1, 1, original_h, original_w, original_d))  # 根据需要调整形状
            show_front_png(final_prediction_reshaped, test_ct, test_pet, test_label, save_dir, step)
            # affine = np.eye(4)  # 根据需要设置仿射变换矩阵
            # nifti_img = nib.Nifti1Image(final_prediction_reshaped, affine)
            # save_file = f"{fold + 1}_prediction_sample_{step + 1}.nii.gz"
            # print(f"save nifti_img \"{save_file}\"")
            # print(f"Fold {fold + 1}: time cost {time.time() - start} s \n")
            # nib.save(nifti_img, os.path.join(save_dir, save_file))



def show_nifti(output, ct, save_dir, epoch, chunk):
    # 可视化保存路径
    save_dir = save_dir + f"show_nifti{epoch + 1}"
    os.makedirs(save_dir, exist_ok=True)  # 创建保存结果的目录

    # 双模态，batch_size 需要除以2
    batch_size = int(output.shape[0] / 2)

    # 去掉batch维度，输出和标签都需要处理
    output = output.view(batch_size*2, 2, *output.shape[2:])  # 恢复为 [batch_size, 2, H, W, D]
    output = output.argmax(dim=1)  # 选择输出通道中的最大值


    # 遍历 batch 内的每个图像进行处理
    for idx in range(batch_size):
        # pet_image = pet_image_batch[idx]
        # ct_image = ct_image_batch[idx]
        # gt_segmentation = gt_segmentation_batch[idx]
        mask = output[idx]
        # 创建一个 Nifti1Image 对象
        nifti_image = nib.Nifti1Image(mask, affine=np.eye(4))  # 使用单位矩阵作为仿射变
        save_path = os.path.join(save_dir, f'sample_chunk_.nii.gz')
        # 保存为 .nii.gz 文件
        nib.save(nifti_image, save_path)
    # # 自动保存结果到文件
    # save_path = os.path.join(save_dir, f'sample_{step + 1}_slice_{slice_idx}.png')
    # plt.savefig(save_path,  dpi=300, bbox_inches='tight')
    # plt.close(fig)  # 关闭当前图像，防止内存泄漏



def sliding_window_crop(image, target_size = (128, 128, 64)):
    b, c, h, w, d = image.shape

    # 计算需要填充的大小
    pad_h = (target_size[0] - (h % target_size[0])) % target_size[0]  # 对于高度
    pad_w = (target_size[1] - (w % target_size[1])) % target_size[1]  # 对于宽度
    pad_d = (target_size[2] - (d % target_size[2])) % target_size[2]  # 对于深度

    # 填充图像
    padded_image = F.pad(image, (0, pad_d, 0, pad_w, 0, pad_h), mode='constant', value=0)
    # print(padded_image.shape)
    patches = []

    # 裁剪补丁
    for b_idx in range(b):
        for z in range(0, padded_image.shape[4] - target_size[2] + 1, target_size[2]):
            for y in range(0, padded_image.shape[3] - target_size[1] + 1, target_size[1]):
                for x in range(0, padded_image.shape[2] - target_size[0] + 1, target_size[0]):
                    patch = padded_image[b_idx:b_idx + 1, :,
                                         x:x + target_size[0],
                                         y:y + target_size[1],
                                         z:z + target_size[2]]
                    # print(patch.shape)
                    patches.append(patch)

    return patches



def ModelsVisual(models_predictions, gt_segmentation, ct_slice):
    # models_predictions = [ours_pred, vnet_pred, swinunetr_pred, nnunet_pred, nestedformer_pred, a2fseg_pred]

    # 自动选择最具代表性的切片
    slice_idx = np.argmax(np.sum(gt_segmentation, axis=(0, 1)))

    fig, axes = plt.subplots(1, len(models_predictions), figsize=(15, 5))

    for i, pred in enumerate(models_predictions):
        # 将每个模型的预测转换为NumPy数组
        mask = pred.cpu().numpy()

        # 计算真阳性、假阴性、假阳性
        true_positive = np.logical_and(mask == 1, gt_segmentation == 1)
        false_negative = np.logical_and(mask == 0, gt_segmentation == 1)
        false_positive = np.logical_and(mask == 1, gt_segmentation == 0)

        # 可视化每个模型的预测结果
        axes[i].imshow(ct_slice, cmap='gray')
        axes[i].imshow(np.ma.masked_where(true_positive[:, :, slice_idx] == 0, true_positive[:, :, slice_idx]),
                       alpha=0.5, cmap='Reds')
        axes[i].imshow(np.ma.masked_where(false_negative[:, :, slice_idx] == 0, false_negative[:, :, slice_idx]),
                       alpha=0.5, cmap='Blues')
        axes[i].imshow(np.ma.masked_where(false_positive[:, :, slice_idx] == 0, false_positive[:, :, slice_idx]),
                       alpha=0.5, cmap='Yellows')
        axes[i].axis('off')

    axes[0].set_title('Ours')
    axes[1].set_title('VNet')
    axes[2].set_title('SwinUNETR')
    axes[3].set_title('nnUNet')
    axes[4].set_title('NestedFormer')
    axes[5].set_title('A2FSeg')

    plt.show()


def test(models, save_dir, ct_path, pet_path, label_path):
    print("Starting Multi-Models Visualizing")

    patch_size =  (128, 128, 64)
    num_channels = [8, 16, 32, 64, 128]
    window_size = [8, 8, 4]
    batch_size = 2
    num_workers = 0

    # setting of model's input
    in_channels = 1
    n_classes = 2

    # dataset split
    split = 0.6
    split_1 = 0.8

    images_CT = sorted(glob(os.path.join(ct_path, "*0000.nii.gz")))
    images_PET = sorted(glob(os.path.join(pet_path, "*0001.nii.gz")))
    labels = sorted(glob(os.path.join(label_path, "*.nii.gz")))


    # data transform
    # test_transforms = Compose(
    #     [
    #         LoadImaged(keys=["img", "img_ct", "seg"], image_only=True),
    #         EnsureChannelFirstd(keys=["img", "img_ct", "seg"]),
    #         # CropForegroundd(keys=["img", "img_ct", "seg"], source_key="img"),
    #         ToTensord(keys=["img", "img_ct"]),
    #     ]
    # )
    train_transforms = Compose(
            [
                LoadImaged(keys=["img", "img_ct", "seg"], image_only=True),
                EnsureChannelFirstd(keys=["img", "img_ct", "seg"]),
                CropForegroundd(keys=["img","img_ct","seg"], source_key="img", select_fn=lambda x:x>x.min()),
                RandCropByPosNegLabeld(
                        keys=["img","img_ct","seg"],
                        label_key="seg",
                        spatial_size=patch_size,
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                # RandRotated(keys=['img', 'img_ct', 'seg'], prob=0.8, range_z=45),
                ToTensord(keys=["img", "img_ct"]),
            ]
        )

    val_transforms = Compose(
            [
                LoadImaged(keys=["img", "img_ct", "seg"], image_only=True),
                EnsureChannelFirstd(keys=["img", "img_ct", "seg"]),
                CropForegroundd(keys=["img","img_ct","seg"], source_key="img", select_fn=lambda x:x>x.min()),

                RandCropByPosNegLabeld(
                    keys=["img","img_ct","seg"],
                    label_key="seg",
                    spatial_size=patch_size,
                    pos=1,
                    neg=0,
                    num_samples=1,
                ),
                ToTensord(keys=["img", "img_ct"]),
            ]
        )

    length = len(images_CT)

    # split dataset
    train_images = images_PET[: int(split * length)]
    train_ct_images = images_CT[: int(split * length)]
    train_labels = labels[: int(split * length)]

    test_images = images_PET[int(split * length):]
    test_ct_images = images_CT[int(split * length):]
    test_labels = labels[int(split * length):]

    train_files = [{"img": img, "img_ct": ct,"seg": seg} \
                    for img, ct, seg in zip(train_images, train_ct_images, train_labels)]
    val_files = [{"img": img, "img_ct": ct,"seg": seg} \
                    for img, ct, seg in zip(test_images, test_ct_images, test_labels)]
    print("Training set includes:", len(train_images))
    print("Validation set includes:", len(test_images))

    # create dataloader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_ds = monai.data.Dataset(data=val_files+train_files, transform=val_transforms)
    test_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # 可视化保存路径
    save_dir = save_dir + f"/visual_AutoPET/"
    os.makedirs(save_dir, exist_ok=True)  # 创建保存结果的目录

    model = H2ASeg(
        patch_size=patch_size,
        in_channels=in_channels,
        n_classes=n_classes,
        num_channels=num_channels,
        window_size=window_size).cuda()
    # heckpoint = torch.load(model_path[1])
    # model.load_state_dict(checkpoint)
    # print(f'load checkpoint from {model_path[1]}\n')
    # model.load_state_dict(torch.load("/home/kasm-user/H2ASeg-main/save/hecktor2022_H2ASeg/11_29/val_best.pth"))
    # model.cuda()
    model.eval()

    # 遍历测试数据
    with torch.no_grad():
        for step, test_data in enumerate(test_loader):
            # 处理当前 batch 数据
            batch_size = int(test_data["img_ct"].shape[0])
            pet = test_data["img"].type(torch.FloatTensor).cuda()
            ct = test_data["img_ct"].type(torch.FloatTensor).cuda()
            labels = test_data["seg"].type(torch.FloatTensor).cuda()

            # 初始化存储每个模型的预测结果
            all_predictions = []

            # 遍历每个模型权重，进行预测
            for model_path in models:
                # create model and load pretrained weight
                # model = H2ASeg(
                #     patch_size=patch_size,
                #     in_channels=in_channels,
                #     n_classes=n_classes,
                #     num_channels=num_channels,
                #     window_size=window_size).cuda()

                # 使用DataParallel 自动分配到多个GPU上
                # print(torch.cuda.device_count())
                # if torch.cuda.device_count() > 1:
                #     print(f"Let's use {torch.cuda.device_count()} GPU!")
                #     model = torch.nn.DataParallel(model)

                # state_dict = torch.load(model_path[1])
                # new_state_dict = {}
                # for k, v in state_dict.items():
                #     name = 'module.' + k
                #     new_state_dict[name] = v
                # model.load_state_dict(new_state_dict)  # 动态加载模型参数

                # model.load_state_dict(torch.load("/home/kasm-user/H2ASeg-main/save/hecktor2022_H2ASeg/09_29_remove_train_dice_0/val_best.pth"))  # 动态加载模型参数
                # print(torch.load(model_path[1])["encoder.conv1_ct.conv_block.0.weight"])
                # checkpoint = torch.load(model_path[1])
                # model.load_state_dict(checkpoint)
                # print(f'load checkpoint from {model_path[1]}\n')
                # model.cuda()
                model.load_state_dict(torch.load(model_path[1]))
                model.cuda()
                output = model(ct, pet)[0]  # 预测
                # output[output != 0] = 1
                # print(output.shape)
                # num_ones = torch.sum(output == 1).item()
                #
                # print(f'aa中元素为 1 的个数: {num_ones}')
                # output = output.view(batch_size, 2, *output.shape[2:])  # 恢复维度
                # print(output.shape)
                output = output.argmax(dim=1)  # 获取预测类别
                # print(output.shape)
                # num_ones = torch.sum(output == 1).item()
                #
                # print(f's中元素为 1 的个数: {num_ones}')
                all_predictions.append(output.cpu().numpy())  # 将预测保存为 numpy 格式

            # print(pet_image_batch.shape)
            pet_image_batch = pet[:].cpu().numpy()
            ct_image_batch = ct[:].cpu().numpy()
            gt_segmentation_batch = labels[:].cpu().numpy()

            # 创建用于可视化的图像
            fig, axes = plt.subplots(batch_size, len(models) + 3, figsize=(6 * (len(models) + 3), 7 * batch_size),
                                     dpi=300)
            axes = axes.ravel()

            # 遍历 batch 内的每个图像进行处理
            for idx in range(batch_size):
                # 获取图像数据
                # print(pet.shape)
                # print(pet_image.shape)
                pet_image = pet_image_batch[idx]
                # print(pet_image.shape)
                ct_image = ct_image_batch[idx]
                gt_segmentation = gt_segmentation_batch[idx]

                pet_image = pet_image.squeeze(0)
                ct_image = ct_image.squeeze(0)
                gt_segmentation = gt_segmentation.squeeze(0)

                # 自动选择最具代表性的切片
                slice_idx = np.argmax(np.sum(gt_segmentation, axis=(0, 1)))

                # 获取 PET 和 CT 图像的切片
                pet_slice = pet_image[:, :, slice_idx]
                ct_slice = ct_image[:, :, slice_idx]
                gt_slice = gt_segmentation[:, :, slice_idx]

                # 绘制 PET 图像
                axes[idx * (len(models) + 3)].imshow(pet_slice, cmap='gray')
                axes[idx * (len(models) + 3)].axis('off')

                # 绘制 CT 图像
                axes[idx * (len(models) + 3) + 1].imshow(ct_slice, cmap='gray')
                axes[idx * (len(models) + 3) + 1].axis('off')

                # 绘制 Ground Truth
                axes[idx * (len(models) + 3) + 2].imshow(np.where(gt_slice == 0, 0, 1), cmap='gray')
                axes[idx * (len(models) + 3) + 2].axis('off')

                if (idx == 0):
                    axes[idx * (len(models) + 3)].set_title('PET', fontsize=50)
                    axes[idx * (len(models) + 3) + 1].set_title('CT', fontsize=50)
                    axes[idx * (len(models) + 3) + 2].set_title('Ground Truth', fontsize=50)

                # 绘制每个模型的预测结果
                for model_idx, prediction in enumerate(all_predictions):
                    # print(model_idx)
                    # print(len(all_predictions))
                    mask = prediction[idx]
                    # num_ones = torch.sum(mask == 1).item()
                    #
                    # print(f'掩码中元素为 1 的个数: {num_ones}')
                    mask_slice = mask[:, :, slice_idx]

                    # 获取真阳性、假阴性和假阳性区域
                    true_positive = np.logical_and(mask_slice == 1, gt_slice == 1)  # 红色区域
                    false_negative = np.logical_and(mask_slice == 0, gt_slice == 1)  # 蓝色区域
                    false_positive = np.logical_and(mask_slice == 1, gt_slice == 0)  # 黄色区域

                    # 绘制预测结果
                    ax_idx = idx * (len(models) + 3) + 3 + model_idx
                    axes[ax_idx].imshow(ct_slice, cmap='gray')

                    # 使用不同颜色表示不同区域
                    red_cmap = mcolors.ListedColormap(['red'])
                    blue_cmap = mcolors.ListedColormap(['blue'])
                    yellow_cmap = mcolors.ListedColormap(['yellow'])

                    axes[ax_idx].imshow(np.ma.masked_where(true_positive == False, mask_slice), alpha=0.7,
                                        cmap=red_cmap)
                    axes[ax_idx].imshow(np.ma.masked_where(false_negative == False, mask_slice), alpha=0.7,
                                        cmap=blue_cmap)
                    axes[ax_idx].imshow(np.ma.masked_where(false_positive == False, mask_slice), alpha=0.7,
                                        cmap=yellow_cmap)
                    if (idx == 0):
                        axes[ax_idx].set_title(f'{models[model_idx][0]}', fontsize=50)
                    axes[ax_idx].axis('off')

            # 自动保存结果到文件
            # 手动调整子图间距
            plt.subplots_adjust(wspace=0.15, hspace=0.1)  # 调整列间距(wspace)和行间距(hspace)
            # fig.text(0.01, 0.5, 'Clinical', va='center', ha='center', rotation=90, fontsize=50)
            fig.tight_layout(pad=1)  # pad 控制整体图像与边缘的间距
            save_path = os.path.join(save_dir, f"AutoPET_step_{step + 1}_slice_{slice_idx}.png")
            print(f"Save AutoPET_step_{step + 1}_slice_{slice_idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 关闭当前图像，防止内存泄漏


if __name__ == '__main__':
    pretrained_model_fold_path = "save/autopet2024_HADSeg/2025_04_22 07-16/val_best.pth"
    use_pretrained_model_fold = True
    ct_path = "/home/kasm-user/H2ASeg-main/autopet_fdg_tumor/imagesTr"
    pet_path = "/home/kasm-user/H2ASeg-main/autopet_fdg_tumor/imagesTr"
    label_path = "/home/kasm-user/H2ASeg-main/autopet_fdg_tumor/labelsTr"
    images_CT = sorted(glob(os.path.join(ct_path, "*0000.nii.gz")))
    images_PET = sorted(glob(os.path.join(pet_path, "*0001.nii.gz")))
    labels = sorted(glob(os.path.join(label_path, "*.nii.gz")))

    # data_pairs = [{"img": pet, "img_ct": ct, "seg": seg} for pet, ct, seg in zip(images_PET, images_CT, labels)]
    # best_models = []
    # k_folds = 8

    # if use_pretrained_model_fold:
    #     best_models = sorted(glob(os.path.join(pretrained_model_fold_path, "*fold199.pth")))
    #     # print(best_models)
    #     # checkpoint = torch.load(pretrained_model_fold_path)
    #     # model.load_state_dict(checkpoint)
    #     print(f'load checkpoint from {pretrained_model_fold_path}\n')
    #     show_crop2nifti(data_pairs, best_models, k_folds, save_dir="/home/kasm-user/H2ASeg-main/postprocessing")
    # else:
    #     show_crop2nifti(data_pairs, best_models, k_folds, save_dir="/home/kasm-user/H2ASeg-main/postprocessing")


    #
    # if use_pretrained_model_fold:
    #     best_model = os.path.join(pretrained_model_fold_path, "val_best.pth")
    #     # print(best_models)
    #     # checkpoint = torch.load(pretrained_model_fold_path)
    #     # model.load_state_dict(checkpoint)
    #     print(f'load checkpoint from {pretrained_model_fold_path}\n')
    #     show_whole_body(images_CT, images_PET, labels, best_model, save_dir=pretrained_model_fold_path)
    # ct_path = "/home/kasm-user/H2ASeg-main/autopet_fdg_tumor/imagesTr"
    # pet_path = "/home/kasm-user/H2ASeg-main/autopet_fdg_tumor/imagesTr"
    # label_path = "/home/kasm-user/H2ASeg-main/autopet_fdg_tumor/labelsTr"
    # models_path = [["Proposed", "/home/kasm-user/H2ASeg-main/save/hecktor2022_H2ASeg/11_29/val_best.pth"],
    #                ["NestedFormer", "/home/kasm-user/H2ASeg-main/save/hecktor2022_H2ASeg/11_29/89.pth"],
    #                ["nnUNet", "/home/kasm-user/H2ASeg-main/save/hecktor2022_H2ASeg/11_29/129.pth"],
    #                ["SwinUNETR", "/home/kasm-user/H2ASeg-main/save/hecktor2022_H2ASeg/11_29/79.pth"],
    #                ["VNet", "/home/kasm-user/H2ASeg-main/save/hecktor2022_H2ASeg/11_29/39.pth"],
    #                ["A2FSeg", "/home/kasm-user/H2ASeg-main/save/hecktor2022_H2ASeg/11_29/19.pth"]]
    # save_dir = r"/home/kasm-user/H2ASeg-main/save"
    # test(models_path, save_dir, ct_path, pet_path, label_path)