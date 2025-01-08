import glob
import os

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


class NonLinear(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(NonLinear, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_ch, output_ch),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(output_ch),
        )

    def forward(self, x):
        return self.main(x)


class InputTNet(nn.Module):
    def __init__(self, num_points):
        super(InputTNet, self).__init__()
        self.num_points = num_points

        self.transT = nn.Sequential(
            NonLinear(3, 64), NonLinear(64, 128), NonLinear(128, 1024)
        )
        self.fc = nn.Sequential(
            NonLinear(1024, 512), NonLinear(512, 256), nn.Linear(256, 9)
        )

    def forward(self, input_data):
        # input_data: (B, N, 3)
        B, N, _ = input_data.shape
        x = input_data.reshape(B * N, 3)
        x = self.transT[0](x)  # 3->64
        x = self.transT[1](x)  # 64->128
        x = self.transT[2](x)  # 128->1024
        x = x.view(B, N, 1024).transpose(1, 2)  # (B,1024,N)
        x = torch.max(x, 2)[0]  # max pool -> (B,1024)
        x = self.fc[0](x)
        x = self.fc[1](x)
        x = self.fc[2](x)  # -> (B,9)

        matrix = x.view(-1, 3, 3)
        out = torch.matmul(input_data, matrix)  # (B,N,3)
        return out


class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNet, self).__init__()
        self.num_points = num_points

        self.featT = nn.Sequential(
            NonLinear(64, 64), NonLinear(64, 128), NonLinear(128, 1024)
        )
        self.fc = nn.Sequential(
            NonLinear(1024, 512), NonLinear(512, 256), nn.Linear(256, 64 * 64)
        )

    def forward(self, input_data):
        # input_data: (B, N, 64)
        B, N, _ = input_data.shape
        x = input_data.reshape(B * N, 64)
        x = self.featT[0](x)
        x = self.featT[1](x)
        x = self.featT[2](x)  # -> (B*N,1024)
        x = x.view(B, N, 1024).transpose(1, 2)  # (B,1024,N)
        x = torch.max(x, 2)[0]  # (B,1024)
        x = self.fc[0](x)
        x = self.fc[1](x)
        x = self.fc[2](x)  # (B,4096)
        matrix = x.view(-1, 64, 64)
        out = torch.matmul(input_data, matrix)  # (B,N,64)
        return out


class PointNetSeg(nn.Module):
    def __init__(self, num_points, num_classes):
        super(PointNetSeg, self).__init__()
        self.num_points = num_points
        self.num_classes = num_classes

        self.input_trans = InputTNet(num_points)
        self.feature_trans = FeatureTNet(num_points)

        self.mlp1 = NonLinear(3, 64)
        self.mlp2 = NonLinear(64, 64)

        self.mlp3 = NonLinear(64, 64)
        self.mlp4 = NonLinear(64, 128)
        self.mlp5 = NonLinear(128, 1024)

        self.seg_mlp1 = NonLinear(1088, 512)
        self.seg_mlp2 = NonLinear(512, 256)
        self.seg_mlp3 = NonLinear(256, 128)
        self.seg_mlp4 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        B, N, _ = x.shape
        x = self.input_trans(x)  # (B,N,3)
        x = self.mlp1(x.view(B * N, 3))
        x = self.mlp2(x)
        x = x.view(B, N, 64)
        local_feat_64 = x
        x = self.feature_trans(x)  # (B,N,64)
        x = self.mlp3(x.view(B * N, 64))
        x = self.mlp4(x)
        x = self.mlp5(x)  # (B*N,1024)
        x = x.view(B, N, 1024)
        global_feat = torch.max(x, 1)[0]  # (B,1024)
        global_feat_expanded = global_feat.unsqueeze(1).repeat(1, N, 1)  # (B,N,1024)
        seg_feat = torch.cat([local_feat_64, global_feat_expanded], dim=2)  # (B,N,1088)
        seg_feat = seg_feat.view(B * N, 1088)
        seg_feat = self.seg_mlp1(seg_feat)
        seg_feat = self.seg_mlp2(seg_feat)
        seg_feat = self.seg_mlp3(seg_feat)
        seg_feat = self.seg_mlp4(seg_feat)  # (B*N,num_classes)
        seg_feat = seg_feat.view(B, N, self.num_classes)
        return seg_feat


class SegmentationDataset(Dataset):
    def __init__(self, root="dataset", num_points=200000, transform=None):
        self.root = root
        self.num_points = num_points
        self.transform = transform

        self.ply_files = sorted(glob.glob(os.path.join(self.root, "data_*.ply")))
        self.annotation_files = [
            f.replace(".ply", "_annotations.npy") for f in self.ply_files
        ]

        # num_classesをアノテーションから計算
        all_labels = []
        print("Scanning dataset for classes...")
        for anno_file in self.annotation_files:
            labels = np.load(anno_file)
            all_labels.append(labels)
        all_labels = np.concatenate(all_labels)
        self.unique_classes = np.unique(all_labels)
        self.num_classes = len(self.unique_classes)
        print(f"Found {self.num_classes} unique classes: {self.unique_classes}")

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        ply_file = self.ply_files[idx]
        anno_file = self.annotation_files[idx]

        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points, dtype=np.float32)  # (M,3)
        labels = np.load(anno_file).astype(np.int64)  # (M,)

        M = points.shape[0]
        # 点数が非常に多い場合、self.num_pointsを増やして可能な限り多くの点をサンプル
        # 全点取得が不可能な場合はランダムサンプリング
        if M > self.num_points:
            choice = np.random.choice(M, self.num_points, replace=False)
        else:
            # num_pointsがMより大きい場合は、重複有りサンプリング
            choice = np.random.choice(M, self.num_points, replace=True)
        points = points[choice, :]
        labels = labels[choice]

        if self.transform:
            points = self.transform(points)

        return points, labels


def main():
    # 大量点数を扱うためにnum_pointsを増やす
    # 例：200000点をサンプル。メモリとGPU能力に応じて調整してください。
    num_points = 200000
    root_dir = "dataset"

    # cudaが使える場合は使う
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # dataset初期化時にnum_classesを計算
    temp_dataset = SegmentationDataset(root=root_dir, num_points=num_points)
    num_classes = temp_dataset.num_classes

    # 再度datasetを同じパラメータで確定（あまり意味ないが明確化）
    dataset = temp_dataset
    batch_size = 2  # 超大規模点数の場合、batch_sizeは小さくするのが現実的
    epochs = 10

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model = PointNetSeg(num_points, num_classes).to(device)
    model = model.float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 学習率は適宜調整

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (points, labels) in enumerate(dataloader):
            points = points.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(points)  # (B,N,C)
            outputs = outputs.permute(0, 2, 1).contiguous()  # (B,C,N)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(
                f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
            )

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "pointnet_seg_large_weights.pth")
    print("Training finished and model saved.")


if __name__ == "__main__":
    main()
