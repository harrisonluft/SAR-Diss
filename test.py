import torch
import os
import glob
import numpy as np
from skimage import io
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import Dataset

dataDir = './Images/Patches'
x_testDir = os.path.join(dataDir, 'test_input/CZU ablation')
y_testDir = os.path.join(dataDir, 'test_mask/CZU')

DEVICE = 'cuda'

# data transformations
transforms = albu.Compose([
    # albu.Normalize(
    #     mean=mean,
    #     std=std,
    # ),
    ToTensorV2()
])

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Accuracy(threshold=0.5),
    smp.utils.metrics.Recall(threshold=0.5),
    smp.utils.metrics.Precision(threshold=0.5),
    smp.utils.metrics.Fscore(threshold=0.5)
]

# test dataset
test_dataset = Dataset(
    x_testDir,
    y_testDir,
    transform=transforms
)

test_dataloader = DataLoader(test_dataset)

best_model = torch.load('./Models/model_19.pth')

test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)

# getting files names for prediction
file_string = "2*.tif"
path = os.path.join(y_testDir, file_string)
files = glob.glob(path)
print(files)
names = [os.path.basename(x) for x in files]

for i in range(len(names)):

    image, gt_mask = test_dataset[i]
    gt_mask = gt_mask.squeeze()
    x_tensor = image.to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    io.imsave('./Images/Patches/pred_mask/CZU Ablation/'+names[i], pr_mask);
