import torch
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import Dataset

# importing mean and std normalization values
mean_fp = './Images/mean.csv'
std_fp = './Images/mean.csv'
mean = np.genfromtxt(mean_fp, delimiter=',')
std = np.genfromtxt(std_fp, delimiter=',')

# input data directories
dataDir = './Images/Patches'
x_trainDir = os.path.join(dataDir,'train_input/CZU_ablation')
y_trainDir = os.path.join(dataDir,'train_mask/CZU')
x_validDir = os.path.join(dataDir,'valid_input/CZU_ablation')
y_validDir = os.path.join(dataDir,'valid_mask/CZU')

# models directories
# modelDir = './Models'
# currentModel = 'model_17.pth'
# checkpoint = torch.load(os.path.join(modelDir,currentModel))

# defining model parameters
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    activation=ACTIVATION,
    in_channels=3,
    classes=1,
)
# load previous model if continuing training
# model.load_state_dict(checkpoint.state_dict()) # loading dict of last training for val/training observation

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    # smp.utils.metrics.Accuracy(threshold=0.5),
    # smp.utils.metrics.Recall(threshold=0.5),
    # smp.utils.metrics.Precision(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.001),
])

# normalize = transforms.Normalize(mean=mean, std=std)

# data transformations
transformations = albu.Compose([

    albu.HorizontalFlip(p=0.5),
    # # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
    albu.GaussNoise(p=0.2),
    # albu.Perspective(p=0.5),
    ToTensorV2(),
    # normalizing between 0-1 via min-max scaler in dataset
])

val_transformations = albu.Compose([
    ToTensorV2(),
    # normalizing between 0-1 via min-max scaler in dataset
])

# Training dataset initialize

train_dataset = Dataset(
    x_trainDir,
    y_trainDir,
    transform=transformations # transpose, normalize and geometric and noise augmentations
)


valid_dataset = Dataset(
    x_validDir,
    y_validDir,
    transform=val_transformations # just transpose and normalize
)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=0)

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# adapted from https://stackoverflow.com/questions/48123023/how-to-appropriately-plot-the-losses-values-acquired-by-loss-curve-from-mlpcl
def draw_result(lst_iter, lst_loss, lst_acc, title):
    plt.plot(lst_iter, lst_loss, '-b', label='Training')
    plt.plot(lst_iter, lst_acc, '-r', label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.legend(loc='upper left')
    plt.title(title+str(id))
    # save image
    plt.savefig("./Models/training curves/"+title+str(id)+".png")  # should before show method
    # show
    plt.show()

if __name__ == '__main__':

# MAKE SURE YOU CHANGE THE MODEL ID
    id = 19
    max_score = 0
    train_DL = []
    train_IOU = []
    valid_DL = []
    valid_IOU = []

    epochs = 200
    for i in range(epochs):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_DL.append(train_logs['dice_loss'])
        train_IOU.append(train_logs['iou_score'])
        valid_DL.append(valid_logs['dice_loss'])
        valid_IOU.append(valid_logs['iou_score'])

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './Models/model_'+str(id)+'.pth')
            print('Model saved')

    draw_result(range(epochs), train_DL, valid_DL, 'Dice Loss_')
    draw_result(range(epochs), train_IOU, valid_IOU, 'IOU_')
    rows = zip(train_DL,valid_DL,train_IOU,valid_IOU)

    with open("./Models/CZU ablation Losses.csv", "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


