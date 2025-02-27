import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


def data_sampler(batch_size, num_points):
    half_batch_size = int(batch_size / 2)
    normal_sampled = torch.randn(half_batch_size, num_points, 3)
    uniform_sampled = torch.rand(half_batch_size, num_points, 3)
    normal_labels = torch.ones(half_batch_size)
    uniform_labels = torch.zeros(half_batch_size)

    input_data = torch.cat((normal_sampled, uniform_sampled), dim=0)
    labels = torch.cat((normal_labels, uniform_labels), dim=0)

    data_shuffle = torch.randperm(batch_size)

    return input_data[data_shuffle].view(-1, 3), labels[data_shuffle].view(-1, 1)


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class PcdNormalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample = sample[:, 0:3]
        centroid = torch.mean(sample, axis=0)
        sample = sample - centroid
        m = torch.max(torch.sqrt(torch.sum(sample**2, axis=1)))
        sample = sample / m
        return sample


class PcdDataset(Dataset):
    def __init__(
        self,
        data_num=512,
        npoints=1024,
        transform=PcdNormalize(),
        cache_dir="./data/",
        for_test=False,
    ):
        self.npoints = npoints
        self.data_num = data_num
        self.transform = transform

        data_cache_path = Path(cache_dir + "data.pt")
        label_cache_path = Path(cache_dir + "label.pt")
        if not for_test and data_cache_path.exists() and label_cache_path.exists():
            self.input_data = torch.load(data_cache_path)
            self.labels = torch.load(label_cache_path)
            return

        class_0_pcd = np.loadtxt("./data/Moebius.txt").astype(np.float32)
        class_1_pcd = np.loadtxt("./data/Zetton.txt").astype(np.float32)
        print("============pcd files loaded")
        print(class_0_pcd.shape)
        pcds_0 = np.empty((0, npoints, class_0_pcd.shape[1]))
        pcds_1 = np.empty((0, npoints, class_1_pcd.shape[1]))
        print("============random sampling")
        for i in range(self.data_num // 4):
            random_idx_0 = random.sample(range(len(class_0_pcd)), self.npoints)
            pcds_0 = np.append(pcds_0, np.array([class_0_pcd[random_idx_0]]), axis=0)
            random_idx_1 = random.sample(range(len(class_1_pcd)), self.npoints)
            pcds_1 = np.append(pcds_1, np.array([class_1_pcd[random_idx_1]]), axis=0)
            if for_test:
                # 可視化のため一時的に
                np.savetxt("./data/test/test_0_0_random.txt", class_0_pcd[random_idx_0])
                np.savetxt("./data/test/test_2_1_random.txt", class_1_pcd[random_idx_1])

        print("============farthest point sampling")
        for i in range(self.data_num // 4):
            sampled_0 = farthest_point_sample(class_0_pcd, self.npoints)
            sampled_1 = farthest_point_sample(class_1_pcd, self.npoints)
            pcds_0 = np.append(pcds_0, np.array([sampled_0]), axis=0)
            pcds_1 = np.append(pcds_1, np.array([sampled_1]), axis=0)
            if for_test:
                # 可視化のため一時的に
                np.savetxt("./data/test/test_1_0.txt", sampled_0)
                np.savetxt("./data/test/test_3_1.txt", sampled_1)

        print("============create label")
        labels_0 = torch.zeros(len(pcds_0))
        labels_1 = torch.ones(len(pcds_1))
        class_0_tensor = torch.from_numpy(pcds_0)
        class_1_tensor = torch.from_numpy(pcds_1)
        self.input_data = torch.cat(
            (class_0_tensor[:, :, 0:3], class_1_tensor[:, :, 0:3]), dim=0
        )
        self.labels = torch.cat((labels_0, labels_1), dim=0)
        # dataset = torch.cat((self.input_data, self.labels), dim=1)
        if not for_test:
            torch.save(self.input_data, data_cache_path)
            torch.save(self.labels, label_cache_path)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        data = self.input_data[index]
        label = self.labels[index]

        if self.transform:
            data = self.transform(data)

        return data.view(-1, 3), label.view(-1, 1)


def main():
    data_set = PcdDataset(32)
    dataloader = torch.utils.data.DataLoader(data_set, batch_size=2, shuffle=True)

    for i in dataloader:
        print(i)


if __name__ == "__main__":
    main()
