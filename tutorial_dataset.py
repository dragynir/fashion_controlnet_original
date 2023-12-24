import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.image_dir = opt.image_folder
        self.df_path = opt.df_path
        self.width = opt.width
        self.height = opt.height
        self.data = pd.read_csv(self.df_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = self.get_prompt(item)

        target = Image.open(os.path.join(self.image_dir, item['ImageId'])).convert("RGB")
        target = target.resize((self.width, self.height), resample=Image.BICUBIC)
        # Normalize target images to [-1, 1].
        target = (np.array(target).astype(np.float32) / 127.5) - 1.0

        mask = np.zeros(
            (len(item["EncodedPixels"]), self.width, self.height), dtype=np.uint8
        )

        labels = []
        for m, (annotation, label) in enumerate(
                zip(item["EncodedPixels"], item["labels"])
        ):
            sub_mask = self.rle_decode(
                annotation, (item["Height"], item["Width"])
            )
            sub_mask = Image.fromarray(sub_mask)
            sub_mask = sub_mask.resize(
                (self.width, self.height), resample=Image.BICUBIC
            )
            mask[m, :, :] = sub_mask
            labels.append(int(label) + 1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        final_label = np.zeros((self.width, self.height), dtype=np.uint8)
        first_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        second_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        third_channel = np.zeros((self.width, self.height), dtype=np.uint8)

        upperbody = [0, 1, 2, 3, 4, 5]
        lowerbody = [6, 7, 8]
        wholebody = [9, 10, 11, 12]

        for i in range(len(labels)):
            if labels[i] in upperbody:
                first_channel += new_masks[i]
            elif labels[i] in lowerbody:
                second_channel += new_masks[i]
            elif labels[i] in wholebody:
                third_channel += new_masks[i]

        first_channel = (first_channel > 0).astype("uint8")
        second_channel = (second_channel > 0).astype("uint8")
        third_channel = (third_channel > 0).astype("uint8")

        final_label = first_channel + second_channel * 2 + third_channel * 3
        conflict_mask = (final_label <= 3).astype("uint8")
        source = (conflict_mask) * final_label + (1 - conflict_mask) * 1

        # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0
        return dict(jpg=target, txt=prompt, hint=source)

    def get_prompt(self, item) -> str:
        return " ".join(list(item['name']))

    def rle_decode(self, mask_rle, shape):
        """
        mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array
        shape: (height,width) of array to return
        Returns numpy array according to the shape, 1 - mask, 0 - background
        """
        shape = (shape[1], shape[0])
        s = mask_rle.split()
        # gets starts & lengths 1d arrays
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1
        # gets ends 1d array
        ends = starts + lengths
        # creates blank mask image 1d array
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        # sets mark pixles
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        # reshape as a 2d mask image
        return img.reshape(shape).T  # Needed to align to RLE direction