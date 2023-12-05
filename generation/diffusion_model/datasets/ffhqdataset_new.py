import os
import cv2
import numpy as np

from torch.utils.data import Dataset


class FFHQDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        set_dir = root_dir
        filenames = []
        self.image_path_list = []
        for sett in os.listdir(set_dir):
            filenames = []
            if sett[-1] != 'p':
                for filename in os.listdir(os.path.join(set_dir, sett)):
                    filenames.append(os.path.join(sett, filename))

                for filename in filenames:
                    filepath = os.path.join(set_dir, filename)
                    self.image_path_list.append(filepath)

        self.transform = transform

        print(f'Dataset Size:{len(self.image_path_list)}')

    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        image = self.load_image(idx)
        label = self.load_label(idx)
        sample = {
            'image': image,
            'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32)
    
    def load_label(self, idx):
        label = np.array(0)

        return label.astype(np.float32)
    
if __name__ == '__main__':
    import os
    import random
    import numpy as np
    # import pdb;pdb.set_trace()
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import sys
    sys.path.append('../')
    import torchvision.transforms as transforms
    from common import Opencv2PIL, TorchRandomHorizontalFlip, TorchMeanStdNormalize, ClassificationCollater, Resize
    
    transform=transforms.Compose([
                                    Opencv2PIL(),
                                    TorchMeanStdNormalize(
                                        mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]),
                                    Resize(32)
                                ])
    P = FFHQDataset('/mnt/home/pangshengyuan/datasets/ffhq/ffhq', transform)
    dataloader = DataLoader(P,
                              batch_size=500,
                              shuffle=True,
                              pin_memory=True,
                              )
    total_img = []
    i = 0
    for data in tqdm(dataloader):
        total_img.extend(data['image'])
        print(data['image'].shape)
        if len(total_img) == 7500:
            torch.save(torch.stack(total_img), 'total_ffhq_train.pt')
            del total_img
            total_img = []
        torch.save(torch.stack(total_img), 'total_ffhq_test.pt')
        
    print(P)