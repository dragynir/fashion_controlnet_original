from dataclasses import dataclass

from tutorial_dataset import MyDataset


@dataclass
class Config:
    image_dir: str
    df_path: str
    width: int
    height: int


config = Config(
    image_dir='./training/iMaterialist(Fashion)/train',
    df_path='../training/iMaterialist(Fashion)/dataset/train.csv',
    width=512,
    height=512,
)


dataset = MyDataset(config)
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
