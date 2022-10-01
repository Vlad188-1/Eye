import torch 
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from torchmetrics import Dice, MeanMetric
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
from pathlib import Path

import os
import configargparse
from loguru import logger
from natsort import natsorted
import json

from utils.utility import SegmentationDataset, get_train_augs, get_val_augs
from utils.loss import CustomLoss


def create_table(path_to_images: str, path_to_masks: str) -> pd.DataFrame:

    all_path_to_images = natsorted([str(p) for p in Path(path_to_images).glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    all_path_to_masks = natsorted([str(p) for p in Path(path_to_masks).glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    assert len(all_path_to_images) == len(all_path_to_masks), "Количество изображений не равно количеству масок"

    df = pd.DataFrame({"images": all_path_to_images, "masks": all_path_to_masks})
    # print(df.head())
    # print(df.values[100])
    # print(len(df))

    # names_bad_images = ["29.jpg", "91.jpg", "126.jpg", "138.jpg", "173.jpg", "253.jpg", "262.jpg",
    #        "277.jpg", "279.jpg", "280.jpg", "284.jpg", "288.jpg", "347.jpg", "349.jpg",
    #        "374.jpg", "559.jpg", "593.jpg", "673.jpg"]

    # bad_images = [os.path.join(str(path_to_images), i) for i in names_bad_images]
    # #bad_images = [path_to_images / i for i in names_bad_images]
    # # print(bad_images)
    # df = df.loc[~df["images"].isin(bad_images)]
    # print(len(df))

    return df    


def main(args):

    torch.manual_seed(1)

    df = create_table(path_to_images=args.path_to_images, path_to_masks=args.path_to_masks)

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=10)

    trainset = SegmentationDataset(train_df, get_train_augs(args))
    validset = SegmentationDataset(valid_df, get_val_augs())

    print(f"Size of Trainset : {len(trainset)}")
    print(f"Size of Validset : {len(validset)}")

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch_size)

    model = smp.Unet(encoder_name=args.encoder, 
                            encoder_weights="imagenet",
                            in_channels=3,
                            classes=1,
                            activation=None)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    loss = CustomLoss()
    metric = Dice().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)


    best_dice = 0
    history = {
      'epochs': [],
      'train_loss': [],
      'train_dice': [],
      'test_loss': [],
      'test_dice': [],
      'lr': []
    }

    model.to(DEVICE)

    for epoch in range(args.epochs):
      
        history['epochs'].append(epoch+1)
      
      # Train epoch
        model.train()
        mean_loss = MeanMetric().to(DEVICE)
        mean_dice = MeanMetric().to(DEVICE)
        
        for x, y in tqdm(trainloader, desc=f"Train epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
          
            optimizer.zero_grad()
            pred = model(x)
          
            mean_dice.update(metric(pred, y.long()))

            l = loss(pred, y.long())
            mean_loss.update(l)

            l.backward()
            optimizer.step()
          
        history['train_loss'].append(mean_loss.compute().item())
        history['train_dice'].append(mean_dice.compute().item())
          
        # Test epoch
        model.eval()
        mean_loss = MeanMetric().to(DEVICE)
        mean_dice = MeanMetric().to(DEVICE)
        
        with torch.no_grad():
            
            for x, y in tqdm(validloader, desc=f"Val epoch {epoch+1}"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                mean_dice.update(metric(pred, y.long()))

                l = loss(pred, y.long())
                mean_loss.update(l)
            
            val_loss = mean_loss.compute().item()

            history['test_loss'].append(val_loss)
            history['test_dice'].append(mean_dice.compute().item())

            if best_dice < history['test_dice'][-1]:
                best_dice = history['test_dice'][-1]
                logger.info(
                    f"Saved best model with dice metric {best_dice:.4f} and dice loss {history['test_loss'][-1]:.4f} \n"
                )
                #print(f"Saved best model with dice metric {best_dice:.4f} and dice loss {history['test_loss'][-1]:.4f} \n")
                torch.save(model, 'model_final.pth')
        scheduler.step()
        print(scheduler.get_last_lr())
        torch.cuda.empty_cache()
        history['lr'].append(scheduler.get_last_lr())
        
    with open("history.json", "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    p = configargparse.ArgParser()
    #p.add('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    p.add('--path_to_images', default="", type=str, help='path to images for segmentation')
    p.add('--path_to_masks', default="", type=str, help="path to segmentatiom masks for images")
    p.add('--epochs', default=100, type=int, help="number of epochs")
    p.add('--encoder', default="timm-efficientnet-b5", type=str, help='encoder name for UNet')
    p.add('--batch_size', default=2, type=int, help='batch size for training')
    p.add('--imgsz', default=608, type=int, help='image size for training')
    p.add('--lr', default=0.003, type=float, help="Learning rate for training")

    args = p.parse_args()

    main(args)