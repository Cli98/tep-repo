import os
from parsers.parser import test_parser, apply_parser
from Networks import create_model
from dataloader import CreateDataLoader_bdd
from utils.util import tensor2labelim
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2

"""
Modify this script to allow inference on multiple sequential images. Only provide gt based on original image.
Input: Allocated to folder input/raw
Output: Redirected to gt folder

Load and conduct inference one by one, just to save time on coding.

Folder structure:
Examples
-----source_folder
-----depth
-----pred

python3 inference.py --dataroot ./datasets/Bdd --mode test --gpu 0 --image-type jpg --label-type png --epoch 2 
--Width 640 --Height 384 --batch_size 1
"""

if __name__ == '__main__':
    test_opt = apply_parser(test_parser())
    test_opt.batch_size = 1
    test_opt.istrain = False

    dataset = CreateDataLoader_bdd(test_opt)
    model = create_model(test_opt)
    model.load_networks(test_opt.epoch)
    model.net.eval()

    # if you want to use your own data, please modify rgb_image, depth_image, camParam and use_size correspondingly.
    source_img_folder = os.path.join(".", "examples", "source")
    pred_img_folder = os.path.join(".", "examples", "pred")

    # The individual name for both rgb and dep are same.
    rgb_image_list = [os.path.join(source_img_folder, img) for img in os.listdir(source_img_folder)
                      if img.endswith(test_opt.image_type)]

    use_size = (test_opt.Width, test_opt.Height)  # Multiple of 32
    # Set up prediction folder, if not exist
    if not os.path.exists(os.path.join(pred_img_folder)):
        os.makedirs(pred_img_folder, exist_ok=False)

    for rgb_image_pth in rgb_image_list:
        # When merging data, leave a simple example to generate placeholder
        rgb_image = cv2.cvtColor(cv2.imread(rgb_image_pth), cv2.COLOR_BGR2RGB)
        oriHeight, oriWidth, _ = rgb_image.shape
        oriSize = (oriWidth, oriHeight)

        # resize image to enable sizes divide 32
        rgb_image = cv2.resize(rgb_image, use_size)
        rgb_image = rgb_image.astype(np.float32) / 255
        rgb_image = transforms.ToTensor()(rgb_image).unsqueeze(dim=0)

        with torch.no_grad():
            pred = model.net(rgb_image)
            palet_file = 'datasets/palette.txt'
            impalette = list(np.genfromtxt(palet_file, dtype=np.uint8).reshape(3 * 256))
            pred_img = tensor2labelim(pred, impalette)
            pred_img = cv2.resize(pred_img, oriSize)
            cv2.imwrite(rgb_image_pth.replace("source", "pred"), pred_img)
