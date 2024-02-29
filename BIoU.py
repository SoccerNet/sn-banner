import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
from cv2 import distanceTransform, DIST_L2, DIST_MASK_PRECISE
from tqdm import tqdm
import pandas as pd

nclasses = 3
distances = [1, 3, 5, 10]
# In the repository "out_models", there are one rep per type of model, and for each type of
# model there is at least one rep per specific cfg. List all the cfgs.
modelType = os.listdir("out_models")
models = []
for modelType in modelType:
    modelsArray = os.listdir("out_models/" + modelType)
    for model in modelsArray:
        models.append(modelType + "/" + model + "/" + "non-tta")
        models.append(modelType + "/" + model + "/" + "tta")
print(models)

# Create a dataframe to store the results
# The columns are the different models, and the rows are the different distance (d) values
# The values are the mBIoU
df = pd.DataFrame(columns=models, index=distances)


class SoccernetBIoUDataloader(data.Dataset):
    def __init__(self, gt_dir, pred_dir, transform=None):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.transform = transform
        self.pred_paths = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir)]

    def __len__(self):
        return len(self.pred_paths)

    def __getitem__(self, idx):
        pred_path = self.pred_paths[idx]
        gt_path = os.path.join(self.gt_dir, os.path.basename(pred_path))
        pred = Image.open(pred_path)
        gt = Image.open(gt_path)
        # Convert to tensor
        pred = torch.from_numpy(np.array(pred)).long()
        gt = torch.from_numpy(np.array(gt)).long()
        if self.transform:
            pred = self.transform(pred)
            gt = self.transform(gt)
        return pred, gt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model in models:
    dataset = SoccernetBIoUDataloader(
        "Dataset/Labels/",
        "out_models/" + model + "/",
    )
    print("Number of images in the test set : ", len(dataset))

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    biou = torch.zeros((len(dataset), nclasses), device=device)

    kernel_borders = (
        torch.tensor(
            [[1, 1, 1], [1, -9, 1], [1, 1, 1]],
            dtype=torch.float16,
            device=device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    for d in distances:
        kernel_d_borders = np.ones((d * 2 + 1, d * 2 + 1), dtype=np.uint8)
        kernel_d_borders[d, d] = 0
        kernel_d_borders = (
            torch.tensor(
                distanceTransform(kernel_d_borders, DIST_L2, DIST_MASK_PRECISE),
                dtype=torch.float16,
                device=device,
            )
            .le(d)
            .half()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        for i, (pred, gt) in enumerate(tqdm(dataloader)):
            pred.squeeze_()
            gt.squeeze_()
            pred = pred.to(device)
            gt = gt.to(device)
            gt_one_hot = (
                torch.nn.functional.one_hot(gt, num_classes=nclasses)
                .permute(2, 0, 1)
                .half()
            )
            pred_one_hot = (
                torch.nn.functional.one_hot(pred, num_classes=nclasses)
                .permute(2, 0, 1)
                .half()
            )
            pred_borders = (
                torch.nn.functional.conv2d(
                    1 - pred_one_hot.unsqueeze(1), kernel_borders, padding="same"
                )
                .squeeze()
                .clamp(0, 1)
            )
            gt_borders = (
                torch.nn.functional.conv2d(
                    1 - gt_one_hot.unsqueeze(1), kernel_borders, padding="same"
                )
                .squeeze()
                .clamp(0, 1)
            )
            pred_d_border = (
                torch.nn.functional.conv2d(
                    pred_borders.unsqueeze(1), kernel_d_borders, padding="same"
                )
                .squeeze()
                .clamp(0, 1)
            )
            gt_d_border = (
                torch.nn.functional.conv2d(
                    gt_borders.unsqueeze(1), kernel_d_borders, padding="same"
                )
                .squeeze()
                .clamp(0, 1)
            )
            pred_intersection = torch.logical_and(pred_one_hot, pred_d_border)
            gt_intersection = torch.logical_and(gt_one_hot, gt_d_border)
            numerator = torch.sum(
                pred_intersection.logical_and(gt_intersection), dim=(1, 2)
            )
            denominator = torch.sum(
                pred_intersection.logical_or(gt_intersection), dim=(1, 2)
            )
            biou[i] = numerator / denominator

        mBIoU = biou.nanmean(dim=0).mean().item()
        print(model, d, mBIoU)
        df[model][d] = mBIoU

print(df)
df.to_csv("BIoU.csv")
