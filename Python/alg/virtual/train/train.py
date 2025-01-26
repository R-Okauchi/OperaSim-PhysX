#!/usr/bin/env python
# coding: utf-8

"""
PointNet++ Semantic Segmentation (Unity Y-axis up) 
with Parallel Sub-block Cropping & num_workers
MSG (Multi-Scale Grouping) version + color (RGB) usage
--------------------------------------------------------------------------------
1. 大規模点群(約25~30万点)を「Y軸が上」の想定で x-z 平面をサブブロックに切り出し
2. sub-block の切り出しを concurrent.futures.ProcessPoolExecutor で並列化
3. DataLoader の num_workers を設定してバッチ処理も並列化
4. 学習率スケジューラ (StepLR) やデータ拡張 (Y軸回り回転) はそのまま
5. MSG (Multi-Scale Grouping) を用いた PointNet++ 実装
6. PLY から読み込む際に色情報 (RGB) も使う => 入力は Nx6 (x,y,z,r,g,b)
"""

import os
import glob
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
import multiprocessing

############################################################
# Utility: Sub-block Cropping (x-z plane)
############################################################

def crop_sub_blocks(points, labels, block_size=20.0, stride=10.0, min_points=200):
    """
    (x-z平面) でサブブロックをスライド切り出し
    points: (N,6) => (x,y,z,r,g,b)
    labels: (N,)
    """
    xyz = points[:, :3]  # 座標だけ取り出す(xyz)

    xyz_min = np.min(xyz, axis=0)  # (3,)
    xyz_max = np.max(xyz, axis=0)

    # x-z の範囲を stride ずつスライド
    x_range = np.arange(xyz_min[0], xyz_max[0] + 1e-6, stride)
    z_range = np.arange(xyz_min[2], xyz_max[2] + 1e-6, stride)

    sub_blocks = []
    for x in x_range:
        for z in z_range:
            x_cond = (xyz[:, 0] >= x) & (xyz[:, 0] < x + block_size)
            z_cond = (xyz[:, 2] >= z) & (xyz[:, 2] < z + block_size)
            cond = x_cond & z_cond
            block_points = points[cond]
            block_labels = labels[cond]
            if len(block_points) < min_points:
                continue
            sub_blocks.append((block_points, block_labels))

    return sub_blocks

############################################################
# Utility: Data Augmentation (Y-axis rotation)
############################################################

def augment_point_cloud(points):
    """
    - ランダム回転 (Y軸周り)
    - ランダムスケーリング
    - ジッタ
    
    points: (N,6) => 前半3次元が座標 (x,y,z)、後半3次元が色 (r,g,b)
            ここでは色に対する拡張は行わず、座標のみ回転させる。
    """
    xyz = points[:, :3]
    rgb = points[:, 3:]  # 色はそのまま

    # 1) ランダム回転 (Y軸周り)
    theta = np.random.uniform(0, 2*np.pi)
    rot_mat = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ], dtype=np.float32)
    xyz = xyz @ rot_mat.T

    # 2) ランダムスケーリング
    scale = np.random.uniform(0.9, 1.1)
    xyz *= scale

    # 3) ジッタ (座標のみ)
    sigma = 0.01
    clip = 0.05
    jitter = np.clip(sigma * np.random.randn(*xyz.shape), -clip, clip)
    xyz += jitter.astype(np.float32)

    # 座標と色を再結合
    augmented = np.hstack((xyz, rgb))
    return augmented

############################################################
# Utility: Basic PointNet++ functions (FPS, Grouping, etc.)
############################################################

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    return:
      if idx.dim()==2 => (B, S, C)
      if idx.dim()==3 => (B, S, K, C)
    """
    points_trans = points.permute(0, 2, 1).contiguous()

    if idx.dim() == 2:
        B, S = idx.shape
        _, C, N = points_trans.shape
        idx_expand = idx.unsqueeze(1).expand(B, C, S)   # (B, C, S)
        gathered = torch.gather(points_trans, 2, idx_expand)  # => (B, C, S)
        gathered = gathered.permute(0, 2, 1).contiguous()     # => (B, S, C)
        return gathered
    elif idx.dim() == 3:
        B, S, K = idx.shape
        _, C, N = points_trans.shape
        idx = idx.contiguous()
        idx_flat = idx.view(B, -1)
        idx_expand = idx_flat.unsqueeze(1).expand(B, C, S*K)
        gathered = torch.gather(points_trans, 2, idx_expand)
        gathered = gathered.view(B, C, S, K)
        gathered = gathered.permute(0, 2, 3, 1).contiguous()
        return gathered
    else:
        raise ValueError(f"idx must be 2D or 3D, but got shape {idx.shape}")

def square_distance(src, dst):
    """
    src, dst: (B, N, 3), (B, M, 3)
    return dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0,2,1))
    dist += torch.sum(src**2, -1).unsqueeze(-1)
    dist += torch.sum(dst**2, -1).unsqueeze(1)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    xyz: (B, N, 3)
    new_xyz: (B, S, 3)
    return idx: (B, S, nsample)
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    dist = square_distance(new_xyz, xyz)  # (B, S, N)
    # k近傍を取る (nsample 個)
    group_idx = dist.argsort(dim=-1)[:, :, :nsample]  # (B, S, nsample)
    # 半径を超えた部分は無効化
    mask = dist.gather(dim=-1, index=group_idx) > (radius ** 2)
    group_idx[mask] = N - 1  # 最後のインデックスに飛ばす
    return group_idx

############################################################
# MSG (Multi-Scale Grouping) based Set Abstraction
############################################################

class PointNetSetAbstractionMsg(nn.Module):
    """
    Multi-Scale Grouping (MSG) 版の Set Abstraction
    radii: [r1, r2, ...]
    nsamples: [ns1, ns2, ...] (radii と同じ長さ)
    mlps: List of List. 例: [[32,32,64], [32,48,64]] のようにスケールごとに別のMLP
    in_channels: points (座標以外) の次元数
                 (注意) grouping 時は座標3次元も concat するので、Conv入力は in_channels + 3
    """
    def __init__(self, npoint, radii, nsamples, in_channels, mlps):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.radii = radii
        self.nsamples = nsamples
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        # スケールごとに Conv-BN のモジュールを構築
        for i in range(len(radii)):
            layers = nn.ModuleList()
            bns = nn.ModuleList()
            last_dim = in_channels + 3  # grouped_xyz_norm (3) + additional features
            for out_dim in mlps[i]:
                layers.append(nn.Conv2d(last_dim, out_dim, 1))
                bns.append(nn.BatchNorm2d(out_dim))
                last_dim = out_dim
            self.conv_blocks.append(layers)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        xyz: (B, N, 3)
        points: (B, N, C)  (C = in_channels)
        return:
          new_xyz: (B, npoint, 3)
          new_points: (B, npoint, sum_of_last_mlp)  (各スケール出力をconcat)
        """
        B, N, _ = xyz.shape

        # (1) Farthest Point Sampling
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)

        # (2) スケールごとに Grouping & MLP
        scale_results = []
        for i, radius in enumerate(self.radii):
            nsample = self.nsamples[i]
            # 近傍インデックス
            idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B,npoint,nsample)
            grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, idx)  # (B,npoint,nsample,C)
                # 座標差分(3) + 追加特徴(C) => (B,npoint,nsample, C+3)
                new_features = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                new_features = grouped_xyz_norm

            # Conv2d にかけるため (B, C+3, npoint, nsample) に permute
            new_features = new_features.permute(0, 3, 1, 2)  # (B, C+3, npoint, nsample)

            # MLP (Conv-BN-ReLU...)
            for conv, bn in zip(self.conv_blocks[i], self.bn_blocks[i]):
                new_features = F.relu(bn(conv(new_features)))

            # Pooling (max pool over nsample次元)
            # 最終形状: (B, out_dim, npoint)
            new_features = torch.max(new_features, -1)[0]
            scale_results.append(new_features)

        # (3) スケール結果をチャネル結合 => (B, sum_outdim, npoint)
        new_points = torch.cat(scale_results, dim=1)

        # 転置して (B, npoint, sum_outdim)
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points

############################################################
# Feature Propagation (同じ)
############################################################

class PointNetFeaturePropagation(nn.Module):
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
        """
        xyz1: (B, N1, 3)  -- upsample先
        xyz2: (B, N2, 3)  -- downsample済み
        points1: (B, N1, C1)  -- upsample先の特徴 (skip link)
        points2: (B, N2, C2)  -- downsample側の特徴
        return: (B, N1, out_dim)
        """
        if xyz2 is None:
            # 一番下の層の場合など
            interpolated_points = points2.unsqueeze(2).repeat(1,1,xyz1.shape[1])
        else:
            dist = square_distance(xyz1, xyz2)
            dist, idx = dist.sort(dim=-1)
            dist = dist[:, :, :3]  # k=3
            idx = idx[:, :, :3]    # (B, N1, 3)

            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            # (B,N1,3,C2)
            interpolated_points = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)
            # => (B,N1,C2)
            interpolated_points = interpolated_points.permute(0,2,1)

        if points1 is not None:
            points1 = points1.permute(0,2,1)  # (B,C1,N1)
            new_points = torch.cat([points1, interpolated_points], dim=1)  # (B,C1+C2,N1)
        else:
            new_points = interpolated_points

        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = new_points.permute(0,2,1)
        return new_points

############################################################
# MSG-based PointNet++ Semantic Segmentation Model
############################################################

class PointNet2SemSegMSG(nn.Module):
    """
    MSG 版の Semantic Segmentation モデル例。
    
    入力: (B, N, 6) => (x,y,z,r,g,b)
    出力: (B, N, num_classes)
    下記の構成は一例なので、実際の用途に合わせて layer 数やMLP寸法を調整してください。
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # 下記は例として2スケールずつのMSGを3層行う構成

        # === SA1 ===
        # npoint=1024, radii=[5.0, 10.0], nsample=[32,64], in_channels=6
        # mlps => スケール1用 [16,16,32], スケール2用 [32,32,64]
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radii=[5.0, 10.0],
            nsamples=[32, 64],
            in_channels=3,  
            mlps=[[16,16,32], [32,32,64]]
        )  # 出力合計: 32 + 64 = 96

        # === SA2 ===
        # npoint=256, radii=[20.0, 40.0], nsample=[32,64], in_channels=96
        # mlps => スケール1用 [64,64,128], スケール2用 [64,96,128]
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=256,
            radii=[20.0, 40.0],
            nsamples=[32, 64],
            in_channels=96,
            mlps=[[64,64,128], [64,96,128]]
        )  # 出力合計: 128 + 128 = 256

        # === SA3 ===
        # npoint=64, radii=[80.0, 160.0], nsample=[32,64], in_channels=256
        # mlps => スケール1用 [128,128,256], スケール2用 [128,196,256]
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=64,
            radii=[80.0,160.0],
            nsamples=[32,64],
            in_channels=256,
            mlps=[[128,128,256], [128,196,256]]
        )  # 出力合計: 256 + 256 = 512

        # === FP層 ===
        # fp3: 前層(=SA2出力256) と SA3出力(512) => 合計768
        self.fp3 = PointNetFeaturePropagation(512+256, [256, 256])
        # fp2: 前層(=SA1出力96) と fp3出力(256) => 合計352
        self.fp2 = PointNetFeaturePropagation(256+96, [256, 128])
        # fp1: 入力(xyz+rgb=6) と fp2出力(128) => 合計134
        self.fp1 = PointNetFeaturePropagation(128+3, [128, 128, 128])

        # 最終 conv (分類用)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1   = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz_rgb):
        """
        xyz_rgb: (B, N, 6) => [x,y,z,r,g,b]
        出力: (B, N, num_classes)
        """
        B, N, _ = xyz_rgb.shape

        # xyz / points に分割
        xyz = xyz_rgb[:, :, :3]  # (B,N,3)
        points = xyz_rgb[:, :, 3:]  # (B,N,3) (ここではrgb)

        # === SA1 ===
        l1_xyz, l1_points = self.sa1(xyz, points)  # l1_xyz:(B,1024,3), l1_points:(B,1024,96)
        # === SA2 ===
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B,256,3), (B,256,256)
        # === SA3 ===
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B,64,3), (B,64,512)

        # === FP層 ===
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)   # => (B,256,256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)   # => (B,1024,128)
        l0_points = self.fp1(xyz, l1_xyz, points, l1_points)         # => (B,N,128)

        # 最終分類
        net = l0_points.permute(0, 2, 1)  # (B,128,N)
        net = F.relu(self.bn1(self.conv1(net)))
        net = self.dropout(net)
        net = self.conv2(net)            # (B,num_classes,N)
        net = net.permute(0,2,1)         # (B,N,num_classes)
        return net

############################################################
# Dataset with Parallel sub-block creation
############################################################

def load_ply_with_annotation(ply_path, annotation_path):
    """
    .plyファイルから (x,y,z) と (r,g,b) を読み込み (=> Nx6)
    アノテーション .npy からラベルを読み込み (=> Nx)
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if len(xyz) == 0:
        raise ValueError(f"No points in PLY: {ply_path}")

    # plyに色が含まれていない場合はゼロ埋めでもOK
    if len(pcd.colors) == 0:
        rgb = np.zeros_like(xyz, dtype=np.float32)
    else:
        rgb = np.asarray(pcd.colors, dtype=np.float32)

    # Nx6
    points = np.hstack((xyz, rgb))

    labels = np.load(annotation_path).astype(np.int64)
    if len(labels) != len(points):
        raise ValueError(f"Mismatch: #points={len(points)} vs #labels={len(labels)}")
    return points, labels

def _parallel_load_and_crop(args):
    """
    並列実行用の関数: ply & npy を読み込み、サブブロックに切り出して返す
    """
    ply_path, ann_path, block_size, stride, min_points = args
    points, labels = load_ply_with_annotation(ply_path, ann_path)
    sub_blocks = crop_sub_blocks(points, labels, block_size, stride, min_points)
    print(f"Loaded {ply_path}, #sub-blocks: {len(sub_blocks)}")
    return sub_blocks  # List of (sub_points, sub_labels)

class PointCloudSubBlockDataset(Dataset):
    """
    1つの .ply を複数のサブブロックに分割し、それら全体をデータセットとする。
    (Unity想定, Y軸が上 => x-z plane で分割)
    """
    def __init__(
        self, root_dir, block_size=20.0, stride=10.0, min_points=200,
        mode='train', transform=True
    ):
        super().__init__()
        self.root_dir = root_dir
        self.block_size = block_size
        self.stride = stride
        self.min_points = min_points
        self.transform = transform

        # ply + annotation のペアを収集
        ply_files = sorted(glob.glob(os.path.join(root_dir, "*.ply")))
        data_list = []
        for ply_path in ply_files:
            ann_path = ply_path.replace(".ply", "_annotations.npy")
            if os.path.exists(ann_path):
                data_list.append((ply_path, ann_path))

        # train/val/test 分割 (8:1:1) の例
        n = len(data_list)
        if mode == 'train':
            data_list = data_list[:int(0.8*n)]
        elif mode == 'val':
            data_list = data_list[int(0.8*n):int(0.9*n)]
        else:
            data_list = data_list[int(0.9*n):]

        # 並列実行で sub-block リストを作る
        parallel_args = [(ply, ann, block_size, stride, min_points) for (ply, ann) in data_list]
        self.sub_blocks = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(_parallel_load_and_crop, parallel_args))

        # results は List of List[ (points, labels), ... ]
        for blocks in results:
            self.sub_blocks.extend(blocks)

    def __len__(self):
        return len(self.sub_blocks)

    def __getitem__(self, idx):
        points, labels = self.sub_blocks[idx]

        if self.transform:
            points = augment_point_cloud(points)

        # 例: クラス数 4 を想定しているなら、ここでバリデーション
        # 必要に応じてラベルのマッピング等も行う
        unique_labels = np.unique(labels)
        valid_labels = [0,1,2,3]  # 例: 4クラスだけを想定
        if not set(unique_labels).issubset(valid_labels):
            # 発見されたラベルが想定外の場合は警告/エラーなど
            print(f"Found unknown labels: {unique_labels}")
            # raise ValueError("Invalid label found.")

        points_t = torch.from_numpy(points)   # (N,6)
        labels_t = torch.from_numpy(labels)   # (N,)
        return points_t, labels_t

############################################################
# collate_fn
############################################################

def collate_fn(batch):
    """
    各サブブロックを (B, Nmax, 6) にパディング or サンプリングしてまとめる
    """
    B = len(batch)
    max_points = max([len(b[0]) for b in batch])

    xyzrgb_tensors = []
    label_tensors = []
    for (points, labels) in batch:
        n = len(points)
        if n > max_points:
            idx = np.random.choice(n, max_points, replace=False)
            points = points[idx]
            labels = labels[idx]
            n = max_points

        pad_points = np.zeros((max_points, 6), dtype=np.float32)
        pad_labels = np.zeros((max_points,), dtype=np.int64)

        pad_points[:n, :] = points
        pad_labels[:n] = labels

        xyzrgb_tensors.append(torch.from_numpy(pad_points))
        label_tensors.append(torch.from_numpy(pad_labels))

    xyzrgb_b = torch.stack(xyzrgb_tensors, dim=0)  # (B, max_points, 6)
    label_b = torch.stack(label_tensors, dim=0)    # (B, max_points)
    return xyzrgb_b, label_b

############################################################
# Training & Evaluation
############################################################

def train_one_epoch(model, dataloader, criterion, optimizer, device='cuda'):
    model.train()
    total_loss = 0.0
    batch_losses = []
    for batch_idx, (xyzrgb_b, labels_b) in enumerate(dataloader):
        # xyzrgb_b: (B, N, 6)
        # labels_b: (B, N)
        xyzrgb_b = xyzrgb_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        logits = model(xyzrgb_b)  # => (B,N,num_classes)
        loss = criterion(logits.permute(0,2,1), labels_b)  # CrossEntropy: (B,num_classes,N) vs (B,N)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Train Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        batch_losses.append(loss.item())

    return total_loss / len(dataloader), batch_losses

def eval_one_epoch(model, dataloader, criterion, device='cuda'):
    model.eval()
    total_loss = 0.0
    batch_losses = []
    correct = 0
    total_pts = 0
    with torch.no_grad():
        for batch_idx, (xyzrgb_b, labels_b) in enumerate(dataloader):
            xyzrgb_b = xyzrgb_b.to(device)
            labels_b = labels_b.to(device)

            logits = model(xyzrgb_b)
            loss = criterion(logits.permute(0,2,1), labels_b)
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=-1)  # (B,N)
            correct += (pred == labels_b).sum().item()
            total_pts += labels_b.numel()
            print(f"Val Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            batch_losses.append(loss.item())

    avg_loss = total_loss / len(dataloader)
    acc = correct / total_pts if total_pts > 0 else 0
    return avg_loss, acc, batch_losses

############################################################
# main
############################################################

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # データセットのルートディレクトリ
    root_dir = "../dataset_20250118"

    # ハイパーパラメータ
    num_classes = 7
    block_size = 20.0
    stride = 10.0
    min_points = 200
    batch_size = 8
    max_epoch = 30
    lr = 1e-3
    step_size = 10
    gamma = 0.5
    patience = 5

    # Dataset & DataLoader
    train_dataset = PointCloudSubBlockDataset(
        root_dir=root_dir,
        block_size=block_size,
        stride=stride,
        min_points=min_points,
        mode='train',
        transform=True,
    )
    val_dataset = PointCloudSubBlockDataset(
        root_dir=root_dir,
        block_size=block_size,
        stride=stride,
        min_points=min_points,
        mode='val',
        transform=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False, pin_memory=True
    )
    val_loader   = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False, pin_memory=True
    )

    # MSGモデル
    model = PointNet2SemSegMSG(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ログ記録用
    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    train_batch_losses = []
    val_batch_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epoch):
        print(f"----- Epoch {epoch+1}/{max_epoch} -----")
        train_loss, t_batch_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, v_batch_loss = eval_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        learning_rates.append(scheduler.get_last_lr()[0])
        train_batch_losses.extend(t_batch_loss)
        val_batch_losses.extend(v_batch_loss)
        print(f"[Epoch {epoch+1}/{max_epoch}] "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, LR: {learning_rates[-1]}")
        df_epoch = pd.DataFrame({
            'Epoch': range(1, epoch + 2),
            'Train Loss': train_losses,
            'Val Loss': val_losses,
            'Val Accuracy': val_accuracies,
            'Learning Rate': learning_rates
        })
        df_epoch.to_csv(f"training_log_epoch_{epoch+1}.csv", index=False)
        
        df_train_batch = pd.DataFrame({
            'Train Batch': range(1, len(t_batch_loss) + 1),
            'Train Batch Loss': t_batch_loss
        })
        df_val_batch = pd.DataFrame({
            'Val Batch': range(1, len(v_batch_loss) + 1),
            'Val Batch Loss': v_batch_loss
        })
        df_train_batch.to_csv(f"training_log_train_batch_epoch_{epoch+1}.csv", index=False)
        df_val_batch.to_csv(f"training_log_val_batch_epoch_{epoch+1}.csv", index=False)
        # ここでは簡単に早期終了 (EarlyStopping) の例
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_epoch{epoch+1}.pth")
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 最終モデル保存
    torch.save(model.state_dict(), "pointnet2_semseg_msg_color_final.pth")
    print("Training finished and model saved.")

    # CSVやグラフなどの出力はお好みで
    df_epoch = pd.DataFrame({
        'Epoch': range(1, len(train_losses)+1),
        'Train Loss': train_losses,
        'Val Loss': val_losses,
        'Val Accuracy': val_accuracies,
        'Learning Rate': learning_rates
    })
    df_epoch.to_csv('training_log_msg_color.csv', index=False)
    df_train_batch = pd.DataFrame({
        'Train Batch': range(1, len(train_batch_losses) + 1),
        'Train Batch Loss': train_batch_losses
    })
    df_val_batch = pd.DataFrame({
        'Val Batch': range(1, len(val_batch_losses) + 1),
        'Val Batch Loss': val_batch_losses
    })
    df_train_batch.to_csv('training_log_train_batch_msg_color.csv', index=False)
    df_val_batch.to_csv('training_log_val_batch_msg_color.csv', index=False)

    # グラフ
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(df_epoch['Epoch'], df_epoch['Train Loss'], label='Train Loss')
    plt.plot(df_epoch['Epoch'], df_epoch['Val Loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('Loss over Epochs (MSG+Color)')

    plt.subplot(1,2,2)
    plt.plot(df_epoch['Epoch'], df_epoch['Val Accuracy'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.title('Validation Accuracy over Epochs (MSG+Color)')

    plt.tight_layout()
    plt.savefig('training_curves_msg_color.png')
    plt.show()


if __name__ == "__main__":
    # Windows環境でのマルチプロセス実行のお作法
    multiprocessing.set_start_method('spawn', force=True)
    main()
