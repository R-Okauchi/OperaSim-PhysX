import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import open3d as o3d

def index_points(points, idx):
    # points: (B, N, C), idx: (B, S)
    B = points.shape[0]
    S = idx.shape[1]
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    return torch.gather(points, 1, idx_expanded)  # (B, S, C)

def farthest_point_sample(xyz, npoint):
    # xyz: (B, N, 3), npoint: int
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    # xyz: (B, N, 3), new_xyz: (B, S, 3)
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    group_idx = torch.arange(N, device=xyz.device).view(1,1,N).repeat([B,S,1])
    sqrdists = torch.sum((xyz.unsqueeze(1) - new_xyz.unsqueeze(2))**2, -1)  # (B, S, N)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:,:,:nsample]
    group_idx[group_idx==N] = group_idx[:,:,0].unsqueeze(-1).repeat(1,1,nsample)[group_idx==N]
    return group_idx

def sample_and_group(xyz, npoint, radius, nsample, points=None):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)        # (B, S)
    new_xyz = index_points(xyz, fps_idx)                # (B, S, 3)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, S, nsample)
    grouped_xyz = index_points(xyz, idx)                # (B, S, nsample, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)      # (B, S, nsample, D)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points

class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channels
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points=None):
        new_xyz, new_points = sample_and_group(
            xyz, self.npoint, self.radius, self.nsample, points
        )
        # new_points: (B, S, nsample, C+D)
        new_points = new_points.permute(0, 3, 1, 2)  # (B, C+D, S, nsample)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, -1)[0]  # (B, mlp[-1], S)
        new_xyz = new_xyz
        return new_xyz, new_points

class FeaturePropagation(nn.Module):
    def __init__(self, in_channels, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channels
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        # アップサンプリング用: xyz2, points2 -> xyz1, points1
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1,1,N)
        else:
            dists = torch.sum((xyz1.unsqueeze(2) - xyz2.unsqueeze(1))**2, dim=-1) # (B, N, S)
            dists, idx = dists.sort(dim=-1)
            dists = dists[:,:,:3]
            idx = idx[:,:,:3] 
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2.permute(0,2,1), idx) * weight.unsqueeze(-1), dim=2)
            interpolated_points = interpolated_points.permute(0,2,1)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class PointNet2Seg(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channels=3, mlp=[32,64,128])
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channels=128+3, mlp=[128,128,256])
        self.sa3 = SetAbstraction(npoint=32, radius=0.8, nsample=64, in_channels=256+3, mlp=[256,512,1024])
        
        self.fp3 = FeaturePropagation(in_channels=1024+256, mlp=[256,256])
        self.fp2 = FeaturePropagation(in_channels=256+128, mlp=[256,128])
        self.fp1 = FeaturePropagation(in_channels=128+3, mlp=[128,128,128])
        self.conv_seg = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz, points=None):
        # xyz: (B, N, 3)
        l1_xyz, l1_points = self.sa1(xyz, points)   # (B, 512, 3), (B, 128, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points.permute(0,2,1)) 
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points.permute(0,2,1)) 

        # 逆方向にアップサンプリング
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, points.permute(0,2,1) if points is not None else None, l1_points)

        seg_feat = self.conv_seg(l0_points)   # (B, num_classes, N)
        return seg_feat



class SegmentationDataset(Dataset):
    def __init__(self, root='dataset', num_points=2048, transform=None):
        self.root = root
        self.num_points = num_points
        self.transform = transform
        
        # pcd_*.plyから対応するアノテーションファイルを取得
        self.ply_files = sorted(glob.glob(os.path.join(self.root, 'pcd_*.ply')))
        self.annotation_files = [f.replace('.ply','_annotations.npy') for f in self.ply_files]

        # 本例ではクラスは[0,1,2,3,4,5]の6種類と想定
        self.unique_classes = np.array([0, 1, 2, 3, 4, 5])
        self.num_classes = len(self.unique_classes)
    
    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        ply_file = self.ply_files[idx]
        anno_file = self.annotation_files[idx]

        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points, dtype=np.float32)
        labels = np.load(anno_file).astype(np.int64)
        
        # 点数をnum_pointsにリサンプリング
        M = points.shape[0]
        if M > self.num_points:
            choice = np.random.choice(M, self.num_points, replace=False)
        else:
            choice = np.random.choice(M, self.num_points, replace=True)
        points = points[choice, :]
        labels = labels[choice]

        if self.transform:
            points = self.transform(points)

        # (N, 3), (N,) を返す
        return points, labels

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for points, labels in train_loader:
        points = points.to(device)        # (B, N, 3)
        labels = labels.to(device)        # (B, N)
        optimizer.zero_grad()
        outputs = model(points)          # (B, num_classes, N)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for points, labels in val_loader:
            points = points.to(device)
            labels = labels.to(device)
            outputs = model(points)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    print("Start training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = SegmentationDataset(root='../dataset', num_points=2048)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = PointNet2Seg(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    for epoch in range(5):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_pointnet2.pth")
            print("Best model saved.")

if __name__ == '__main__':
    main()