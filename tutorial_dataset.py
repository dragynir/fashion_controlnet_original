import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
import pandas as pd


class FashionDataset(Dataset):
    """Image dataset for Controlnet training."""
    def __init__(self, opt):
        self.opt = opt
        self.image_dir = opt.image_dir
        self.df_path = opt.df_path
        self.attributes_path = opt.attributes_path
        self.caption_path = opt.caption_path

        self.width = opt.width
        self.height = opt.height
        self.data = self.prepare_dataset(self.df_path, self.attributes_path)

    def __len__(self):
        """Return length of dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Create single datapoint."""
        item = self.data.iloc[idx]

        prompt = self.get_prompt(item)

        target = self.get_target(item)
        source = self.get_source(item)

        return dict(jpg=target, txt=prompt, hint=source)

    def get_source(self, item) -> np.ndarray:
        """Create source image (control image)."""

        mask = np.zeros(
            (len(item["EncodedPixels"]), self.width, self.height), dtype=np.uint8
        )

        labels = []
        for m, (annotation, label) in enumerate(
                zip(item["EncodedPixels"], item["CategoryId"])
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
        # оставляю маску как в https://github.com/levindabhi/cloth-segmentation/tree/main
        # чтобы можно было юзать предобученную сегму

        # Normalize source images to [0, 1]. (У нас четыре класса, поэтому делим на 3
        source = source.astype(np.float32) / 3.0
        # Делаем трехканальное изображение
        source = np.stack([source, source, source], axis=-1)
        return source

    def get_target(self, item) -> np.ndarray:
        """Create target image (otuput)"""
        target = Image.open(os.path.join(self.image_dir, item['ImageId'])).convert("RGB")
        target = target.resize((self.width, self.height), resample=Image.BICUBIC)
        # Normalize target images to [-1, 1].
        target = (np.array(target).astype(np.float32) / 127.5) - 1.0
        return target

    def get_prompt(self, item) -> str:
        """Construct prompt from metadata."""
        return item['fast_clip_prompts']

    def rle_decode(self, mask_rle, shape):
        """Decode mask from annotation.

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

    def prepare_dataset(self, df_path: str, caption_path: str, attributes_path: str) -> pd.DataFrame:
        """Create dataset from raw data."""

        label_description = open(attributes_path).read()
        image_info = json.loads(label_description)

        categories = pd.DataFrame(image_info['categories'])
        attributes = pd.DataFrame(image_info['attributes'])

        train_df = pd.read_csv(df_path)
        caption_df = pd.read_csv(caption_path)

        # find records with attributes
        train_df['hasAttributes'] = train_df.ClassId.apply(lambda x: x.find("_") > 0)

        # get main category
        train_df['CategoryId'] = train_df.ClassId.apply(lambda x: x.split("_")[0]).astype(int)

        # supercategory - это категории по типу(верхняя часть тела, нижняя, голова и т д) - по ней для сегмы надо определить в группы
        # name - тут более подробное описание - что на картинке есть
        train_df = train_df.merge(categories, left_on="CategoryId", right_on="id")

        size_df = train_df.groupby("ImageId")[["Height", "Width"]].mean().reset_index()
        size_df = size_df.astype({'Height': 'int', 'Width': 'int'})

        image_df = (
            train_df.groupby("ImageId")[["EncodedPixels", "CategoryId", "name", "supercategory"]]
            .agg(lambda x: list(x))
            .reset_index()
        )
        image_df = image_df.merge(size_df, on="ImageId", how="left")

        # extract all available attributes and create separate table
        cat_attributes = []
        for i in train_df[train_df.hasAttributes].index:
            item = train_df.loc[i]
            xs = item.ClassId.split("_")
            for a in xs[1:]:
                cat_attributes.append({'ImageId': item.ImageId, 'category': int(xs[0]), 'attribute': int(a)})
        cat_attributes = pd.DataFrame(cat_attributes)

        cat_attributes = cat_attributes.merge(
            categories, left_on="category", right_on="id"
        ).merge(attributes, left_on="attribute", right_on="id", suffixes=("", "_attribute"))

        cat_image_df = (
            cat_attributes.groupby("ImageId")[["name", "name_attribute"]]
            .agg(lambda x: list(x))
            .reset_index()
        )
        named_attributes = []
        for _, row in cat_image_df.iterrows():
            named_attributes.append({k: v for k, v in zip(row['name'], row['name_attribute'])})
        cat_image_df['named_attributes'] = named_attributes

        image_df = image_df.merge(cat_image_df[["ImageId", "named_attributes"]], on="ImageId", how="left")
        image_df = image_df.merge(caption_df, on="ImageId", how="left")

        return image_df
