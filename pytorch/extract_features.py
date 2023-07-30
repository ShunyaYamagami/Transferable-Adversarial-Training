import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='Office31', choices=['Office31', 'OfficeHome'])
parser.add_argument("--task", type=str, default='true_domains')
parser.add_argument("--dset", type=str, default='amazon_dslr')
parser.add_argument("--train_batch_size", type=int, default=512)
args = parser.parse_args()

args.text_path = os.path.join('/nas/data/syamagami/GDA/data/GDA_DA_methods/data', args.dataset, args.task, args.dset)
args.labeled_path = os.path.join(args.text_path, 'labeled.txt')
args.unlabeled_path = os.path.join(args.text_path, 'unlabeled.txt')
args.test_path = os.path.join(args.text_path, 'test.txt')


class ImageDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        lines = open(file_path, 'r').read().splitlines()
        lines = np.array([l.strip('\n').split(' ') for l in lines])
        self.image_paths = lines[:, 0]
        self.labels = torch.from_numpy(lines[:, 1].astype(int))
        self.domains = torch.from_numpy(lines[:, 2].astype(int))
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        domain = self.domains[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, domain
    

def extract_features(file_path, save_pth):
    # DataLoaderの作成
    dataset = ImageDataset(file_path)
    data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)

    # ResNet-50の事前学習モデルをロード
    model = models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)
    model = model.eval()

    # モデルの最終層を削除して特徴ベクトルのみを出力するように変更
    model = torch.nn.Sequential(*list(model.children())[:-1]).cuda()

    # フォワードパスを実行して特徴ベクトルを得る
    features_list = torch.Tensor([])
    classes_list = torch.Tensor([])
    domains_list = torch.Tensor([])
    for image, label, domain in data_loader:
        image = image.cuda()

        with torch.no_grad():
            features = model(image)

        # 特徴ベクトルを1次元に平坦化
        features = features.view(features.size(0), -1)
        features_list = torch.cat([features_list, features.detach().cpu()], dim=0)
        classes_list = torch.cat([classes_list, label], dim=0)
        domains_list = torch.cat([domains_list, domain], dim=0)

    output = [features_list, classes_list, domains_list]
    os.makedirs(str(Path(save_pth).parent), exist_ok=True)
    torch.save(output, save_pth)



if __name__ == '__main__':
    save_dir = f'checkpoints/{args.dataset}/{args.task}/{args.dset}'
    
    save_pth = os.path.join(save_dir, Path(args.labeled_path).stem+'.pth')
    if not os.path.exists(save_pth):
        extract_features(args.labeled_path, save_pth)

    save_pth = os.path.join(save_dir, Path(args.unlabeled_path).stem+'.pth')
    if not os.path.exists(save_pth):
        extract_features(args.unlabeled_path, save_pth)