import torch
from torch.utils.data import DataLoader

import numpy as np
from PIL import ImageOps, Image

import configargparse
from pathlib import Path
from tqdm import tqdm
import os

from utils.utility import SimpleDataset, get_val_augs
from utils.post_processing import delete_small_area


def inference(args):

    model_best = torch.load(args.path_to_weights)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    model_best.eval()
    model_best.to(DEVICE)

    testset = SimpleDataset(args.path_to_images, get_val_augs())
    test_loader = DataLoader(testset, batch_size=1)

    with torch.set_grad_enabled(False):
        for image, name_image, height, width in tqdm(test_loader, desc="Обработка данных: ", colour="green", total=len(test_loader)):

            image = image.to(DEVICE)

            # with torch.no_grad():
            logits_mask = model_best(image)

            pred_mask = torch.sigmoid(logits_mask)
            pred_mask[pred_mask >= 0.5] = 1
            pred_mask[pred_mask < 0.5] = 0
            pred_mask = pred_mask.cpu().numpy().squeeze(0)
            pred_mask = np.transpose(pred_mask, (2,1,0))
            pred_mask = np.squeeze(pred_mask, axis=2)
            pred_mask = (pred_mask*255).astype(np.uint8)
            #pred_mask = np.squeeze(pred_mask, axis=2)

            # Post processing
            pred_mask = delete_small_area(args, pred_mask)
            pred_mask = Image.fromarray(pred_mask, mode="L")
            
            border = (4, 8, 4, 8) # left, top, right, bottom
            pred_mask = ImageOps.crop(pred_mask, border)

            # if Path(args.output_path).exists():
            #     shutil.rmtree(args.output_path)

            Path(args.output_path).mkdir(exist_ok=True)

            final_path = os.path.join(args.output_path, f"{Path(name_image[0]).stem}.png")
            pred_mask.save(final_path)


if __name__ == "__main__":

    p = configargparse.ArgParser()
    p.add('--path_to_images', default="", type=str, help='path to images for inference')
    p.add('--path_to_weights', default="", type=str, help='full path to weights (with .pth file)')
    p.add('--min_size_area', default=200, type=int, help='Minimal size of area')
    p.add('--output_path', default="predicted_masks", type=str, help='Output path for predicte masks')

    args = p.parse_args()

    inference(args)