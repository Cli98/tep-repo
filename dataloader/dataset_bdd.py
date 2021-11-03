import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from parsers.parser import basic_parser


class freespace_bdd(data.Dataset):
    def __init__(self, opt):
        super(freespace_bdd, self).__init__()
        self.image_list = []
        self.root = ""
        self.image_affix = "jpg"
        self.label_affix = "png"
        self.mode = "train"
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_labels = 2
        self.__setitem__(self.opt)

    def __setitem__(self, dict):
        self.root = dict.dataroot
        self.image_affix = dict.image_type
        self.label_affix = dict.label_type
        self.mode = dict.mode
        self.compact_size = (dict.Width, dict.Height)
        assert self.compact_size[0] % 32 == 0 and self.compact_size[
            1] % 32 == 0, "input size should be dividable by factor of 32!"
        if dict.mode == "train":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'training', 'image_2', '*.' + self.image_affix)))
        elif dict.mode == "val":
            self.image_list = sorted(
                glob.glob(os.path.join(self.root, 'validation', 'image_2', '*.' + self.image_affix)))
        else:
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'testing', 'image_2', '*.' + self.image_affix)))
        assert len(self.image_list)>0, "Not able to locate image data!"

    def __getitem__(self, index):
        color_image = cv2.cvtColor(cv2.imread(self.image_list[index]), cv2.COLOR_BGR2RGB)
        useDir = "/".join(self.image_list[index].split('/')[:-2])
        name = self.image_list[index].split('/')[-1]
        height, width, depth = color_image.shape
        label = np.zeros((height, width), dtype=np.uint8)
        if self.mode.lower() in ["train", "val"]:
            if not os.path.exists(
                    os.path.join(useDir, 'gt_image_2', name.replace("." + self.image_affix,
                                                                    "") + "_drivable_color" + "." + self.label_affix)):
                print("label path is : {} \n label does not exist! please "
                      "check whether the path is correct!".format(
                      os.path.join(useDir, 'gt_image_2', name.replace("." + self.image_affix,"") +
                                   "_drivable_color" + "." + self.label_affix)))
                exit(-1)
            gt_image = cv2.cvtColor(
                cv2.imread(os.path.join(useDir, 'gt_image_2', name.replace("." + self.image_affix,
                                                                           "") + "_drivable_color" + "." + self.label_affix)),
                cv2.COLOR_BGR2RGB)
            label[gt_image[:, :, 2] > 0] = 1
            label[gt_image[:, :, 1] > 0] = 1
            label[gt_image[:, :, 0] > 0] = 1

        compact_color_image = cv2.resize(color_image, self.compact_size, interpolation=cv2.INTER_CUBIC)
        compact_color_image = transforms.ToTensor()(compact_color_image)

        compact_label = cv2.resize(label, self.compact_size, interpolation=cv2.INTER_CUBIC)
        compact_label[compact_label > 0] = 1
        compact_label = torch.from_numpy(compact_label)
        compact_label = compact_label.type(torch.LongTensor)

        return {'rgb_image': compact_color_image, 'label': compact_label,
                'path': name, 'oriSize': (width, height)}

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    opt = basic_parser()
    dataset = freespace_bdd(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for item in dataloader:
        color_image, label, path, oriSize = np.transpose(np.squeeze(item["rgb_image"].numpy(), axis=0),
                                                         (1, 2, 0)), np.squeeze(item["label"].numpy(), axis=0), item[
                                                "path"], item["oriSize"]
        plt.imshow(color_image)
        plt.imshow(label, alpha=0.3)
        plt.axis('off')
        plt.show()
        plt.pause(0.1)
        print("The path is: {} \n".format(path[0]))
