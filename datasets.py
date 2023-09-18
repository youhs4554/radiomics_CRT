from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd
import glob
from natsort import natsorted
import os
import sys
import random
from skimage.exposure import rescale_intensity
from sklearn.preprocessing import MinMaxScaler


def init_dataset(args, augmentation=False):

    # for pure CT images are used for training
    normalize_func = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
            0.229, 0.224, 0.225])
    ])

    if augmentation:
        # image transforms
        image_transforms = {
            'train': transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.66, 1.0)),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomHorizontalFlip(),
                normalize_func
            ]),
            'test': transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                normalize_func,
            ]),
        }
    else:
        image_transforms = {
            'train': transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(224),
                normalize_func
            ]),
            'test': transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(224),
                normalize_func,
            ]),
        }

    # train/test dataset & loader
    dataset_root = '/data/radiomics_2/classified/%s/%s/%s' % (
        args.image_mode, args.pixelSize, args.label_name)

    datasets = {
        split: MRDataset(root=dataset_root,
                         args=args,
                         split=split,
                         transform=image_transforms["test" if split in ['test', 'train_fixed'] else "train"]) for split in ['train', 'test', 'train_fixed']
    }

    train_loader = DataLoader(
        datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(
        datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    train_fixed_loader = DataLoader(
        datasets['train_fixed'], batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    dataloaders = dict(zip(['train', 'test', 'train_fixed'], [
                       train_loader, test_loader, train_fixed_loader]))

    return datasets, dataloaders


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices, reverse=False):
        out = frame_indices

        if reverse:
            out = out[::-1]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class MRDataset(Dataset):
    def __init__(self, root, args, split='train', transform=None):
        self.root = root
        self.args = args
        self.transform = transform
        self.target_transform = transform
        self.split = split

        (self.frame_dirs, self.clinical_data,
         self.target_vals), self.max_depth, self.min_depth = self.get_datasets(split)
        self.padding = LoopPadding(size=self.max_depth)

    def get_datasets(self, split, suffix='/*/*'):
        # Cross-validation
        frame_dirs = glob.glob(self.root + suffix)
        frame_dirs = np.array([x for x in frame_dirs if Image.open(
            os.path.join(x, os.listdir(x)[0])).size[0] == self.args.resolution])
        volume_ids = [os.path.basename(s) for s in frame_dirs]

        target_vals = np.array(
            [os.path.dirname(s).split('/')[-1] for s in frame_dirs])

        # for exp.
        intervalidation_df = pd.read_excel(
            "misc/Internalvalidation_DCMNO2__latest.xlsx")
        clinical_data = pd.read_excel(
            "misc/clinical_factor_binary.xlsx"
        )
        dcm_numbers = clinical_data["DCMNO"]
        del clinical_data["DCMNO"]

        # use "T.g", "stage.g", "HPV.g2"
        clinical_data = clinical_data[["T.g", "stage.g", "HPV.g2"]]

        train_ids = intervalidation_df["training dataset"].astype(str)
        test_ids = intervalidation_df["test dataset"].dropna().astype(
            int).astype(str).values

        train_ixs = np.array([volume_ids.index(
            i) for i in train_ids])
        test_ixs = np.array([volume_ids.index(i) for i in test_ids])

        train_frame_dirs, test_frame_dirs = frame_dirs[train_ixs], frame_dirs[test_ixs]
        train_target_vals, test_target_vals = target_vals[train_ixs], target_vals[test_ixs]
        train_clinical_data, test_clinical_data = clinical_data[dcm_numbers.isin(
            train_ids)].values, clinical_data[dcm_numbers.isin(
                test_ids)].values

        # integrate as dataframe
        train_df = pd.DataFrame({
            "frame_dirs": train_frame_dirs,
            **{k: v for k, v in zip(clinical_data.columns, train_clinical_data.T)},
            "target": train_target_vals
        })
        test_df = pd.DataFrame({
            "frame_dirs": test_frame_dirs,
            **{k: v for k, v in zip(clinical_data.columns, test_clinical_data.T)},
            "target": test_target_vals
        })

        # train
        train_target_vals = np.array(train_df.pop("target"))
        train_frame_dirs = np.array(train_df.frame_dirs)
        train_clinical_data = np.array(train_df.iloc[:, 1:4])
        # test
        test_target_vals = np.array(test_df.pop("target"))
        test_frame_dirs = np.array(test_df.frame_dirs)
        test_clinical_data = np.array(test_df.iloc[:, 1:4])

        # scaling of clinical data
        clinical_scaler = MinMaxScaler()

        # clinical
        train_clinical_data = clinical_scaler.fit_transform(
            train_clinical_data)
        test_clinical_data = clinical_scaler.transform(
            test_clinical_data)

        # numpy to list
        train_frame_dirs = train_frame_dirs.tolist()
        test_frame_dirs = test_frame_dirs.tolist()

        data = {'train': [train_frame_dirs, train_clinical_data, train_target_vals],
                'test': [test_frame_dirs, test_clinical_data, test_target_vals]}

        max_depth = max(list(len(os.listdir(x)) for x in frame_dirs))
        min_depth = min(list(len(os.listdir(x)) for x in frame_dirs))

        if split == "train_fixed":
            split = "train"

        return data[split], max_depth, min_depth

    def load_frames(self, frame_root, seed, transform=None):
        filenames = natsorted([x for x in os.listdir(frame_root)])
        filenames = list(filter(lambda x: x != 'DCMs', filenames))

        res = []
        for fn in filenames:
            fn = os.path.join(frame_root, fn)
            img = Image.open(fn).convert('L')

            random.seed(seed)

            if transform is not None:
                img = transform(img)
            else:
                img = TF.to_tensor(img)
            res.append(img.numpy())

        res = np.array(res)
        res = torch.from_numpy(res).permute(1, 0, 2, 3)

        # zero-padding
        return F.pad(res, pad=(
            0, 0,
            0, 0,
            0, self.max_depth-res.size(1)
        ))

    def __len__(self):
        return len(self.frame_dirs)

    def __getitem__(self, ix):
        seed = random.randint(-sys.maxsize, sys.maxsize)
        data = self.load_frames(self.frame_dirs[ix], seed, self.transform)
        clinical_data = torch.tensor(self.clinical_data[ix]).float()

        target = torch.tensor(int(self.target_vals[ix])).long()
        target_onehot = torch.eye(2)[target]

        image_id = os.path.basename(self.frame_dirs[ix])

        if self.args.use_clinical:
            return data, clinical_data, target_onehot, image_id
        else:
            return data, target_onehot, image_id
