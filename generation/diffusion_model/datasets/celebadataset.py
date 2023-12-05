import os
import cv2
import numpy as np

from torch.utils.data import Dataset


class CelebADataset(Dataset):
    '''
    https://github.com/tkarras/progressive_growing_of_gans
    '''

    def __init__(self, root_dir, transform=None):
        set_dir = root_dir
        filenames = []
        for filename in os.listdir(set_dir):
            filenames.append(filename)

        self.image_path_list = []
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
            'label': label,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_label(self, idx):
        label = np.array(0)

        return label.astype(np.float32)

if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)

    from tools.path import CelebAHQ_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.diffusion_model.common import Resize, RandomHorizontalFlip, Normalize, ClassificationCollater

    celebahqtraindataset = CelebAHQDataset(root_dir=CelebAHQ_path,
                                           set_name='train',
                                           transform=transforms.Compose([
                                               Resize(resize=256),
                                               RandomHorizontalFlip(prob=0.5),
                                               Normalize(),
                                           ]))

    count = 0
    for per_sample in tqdm(celebahqtraindataset):
        print(per_sample['image'].shape, per_sample['label'].shape,
              per_sample['label'], type(per_sample['image']),
              type(per_sample['label']))

        # temp_dir = './temp'
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)

        # per_sample['image'] = per_sample['image'] * 255.
        # color = [random.randint(0, 255) for _ in range(3)]
        # image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # label = per_sample['label']
        # text = f'label:{int(label)}'
        # cv2.putText(image,
        #             text, (30, 30),
        #             cv2.FONT_HERSHEY_PLAIN,
        #             1.5,
        #             color=color,
        #             thickness=1)

        # cv2.imencode('.jpg', image)[1].tofile(
        #     os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 10:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = ClassificationCollater()
    train_loader = DataLoader(celebahqtraindataset,
                              batch_size=128,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, labels = data['image'], data['label']
        print(images.shape, labels.shape)
        print(images.dtype, labels.dtype)
        if count < 10:
            count += 1
        else:
            break

    ilsvrc2012valdataset = CelebAHQDataset(root_dir=CelebAHQ_path,
                                           set_name='val',
                                           transform=transforms.Compose([
                                               Resize(resize=256),
                                               Normalize(),
                                           ]))

    count = 0
    for per_sample in tqdm(ilsvrc2012valdataset):
        print(per_sample['image'].shape, per_sample['label'].shape,
              per_sample['label'], type(per_sample['image']),
              type(per_sample['label']))

        if count < 10:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = ClassificationCollater()
    val_loader = DataLoader(ilsvrc2012valdataset,
                            batch_size=128,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collater)

    count = 0
    for data in tqdm(val_loader):
        images, labels = data['image'], data['label']
        print(images.shape, labels.shape)
        print(images.dtype, labels.dtype)
        if count < 10:
            count += 1
        else:
            break