import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import multiprocessing as mp
from functools import partial
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import csv
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
class NonLinear(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(NonLinear, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_ch, output_ch),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(output_ch)
        )

    def forward(self, x):
        return self.main(x)


class InputTNet(nn.Module):
    def __init__(self, num_points):
        super(InputTNet, self).__init__()
        self.num_points = num_points

        self.transT = nn.Sequential(
            NonLinear(3, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024)
        )
        self.fc = nn.Sequential(
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 9)
        )

    def forward(self, input_data):
        # input_data: (B, N, 3)
        B, N, _ = input_data.shape
        x = input_data.reshape(B*N, 3)
        x = self.transT[0](x) # 3->64
        x = self.transT[1](x) #64->128
        x = self.transT[2](x) #128->1024
        x = x.view(B, N, 1024).transpose(1,2) # (B,1024,N)
        x = torch.max(x, 2)[0]  # max pool -> (B,1024)
        x = self.fc[0](x)
        x = self.fc[1](x)
        x = self.fc[2](x) # -> (B,9)
        
        matrix = x.view(-1, 3, 3)
        out = torch.matmul(input_data, matrix) # (B,N,3)
        return out

class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNet, self).__init__()
        self.num_points = num_points

        self.featT = nn.Sequential(
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024)
        )
        self.fc = nn.Sequential(
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 64*64)
        )

    def forward(self, input_data):
        # input_data: (B, N, 64)
        B, N, _ = input_data.shape
        x = input_data.reshape(B*N,64)
        x = self.featT[0](x)
        x = self.featT[1](x)
        x = self.featT[2](x) # -> (B*N,1024)
        x = x.view(B, N, 1024).transpose(1,2) # (B,1024,N)
        x = torch.max(x, 2)[0] # (B,1024)
        x = self.fc[0](x)
        x = self.fc[1](x)
        x = self.fc[2](x)  # (B,4096)
        matrix = x.view(-1,64,64)
        out = torch.matmul(input_data, matrix) # (B,N,64)
        return out

class PointNetSeg(nn.Module):
    def __init__(self, num_points, num_classes):
        super(PointNetSeg, self).__init__()
        self.num_points = num_points
        self.num_classes = num_classes

        self.input_trans = InputTNet(num_points)
        self.feature_trans = FeatureTNet(num_points)

        self.mlp1 = NonLinear(3,64)
        self.mlp2 = NonLinear(64,64)
        
        self.mlp3 = NonLinear(64,64)
        self.mlp4 = NonLinear(64,128)
        self.mlp5 = NonLinear(128,1024)

        self.seg_mlp1 = NonLinear(1088,512)
        self.seg_mlp2 = NonLinear(512,256)
        self.seg_mlp3 = NonLinear(256,128)
        self.seg_mlp4 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        B, N, _ = x.shape
        x = self.input_trans(x) # (B,N,3)
        x = self.mlp1(x.view(B*N,3))
        x = self.mlp2(x)
        x = x.view(B,N,64)
        local_feat_64 = x
        x = self.feature_trans(x) # (B,N,64)
        x = self.mlp3(x.view(B*N,64))
        x = self.mlp4(x)
        x = self.mlp5(x) # (B*N,1024)
        x = x.view(B,N,1024)
        global_feat = torch.max(x,1)[0] #(B,1024)
        global_feat_expanded = global_feat.unsqueeze(1).repeat(1,N,1) #(B,N,1024)
        seg_feat = torch.cat([local_feat_64, global_feat_expanded], dim=2) #(B,N,1088)
        seg_feat = seg_feat.view(B*N,1088)
        seg_feat = self.seg_mlp1(seg_feat)
        seg_feat = self.seg_mlp2(seg_feat)
        seg_feat = self.seg_mlp3(seg_feat)
        seg_feat = self.seg_mlp4(seg_feat) #(B*N,num_classes)
        seg_feat = seg_feat.view(B,N,self.num_classes)
        return 



def load_labels(anno_file):
    """ラベルファイルを読み込む関数"""
    try:
        labels = np.load(anno_file)
        unique_count = len(np.unique(labels))
        print(f"Found {unique_count} unique classes in {anno_file}")
        return labels
    except Exception as e:
        print(f"Error loading {anno_file}: {e}")
        return None

def voxelize_point_cloud_worker(voxel_point, tree, labels):
    dist, idx = tree.query(voxel_point)
    return labels[idx]
class SegmentationDataset(Dataset):
    def __init__(self, root='dataset', num_points=200000, voxel_size=0.05, transform=None):
        self.root = root
        print(f"Loading dataset from {root}")
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.transform = transform
        
        self.ply_files = sorted(glob.glob(os.path.join(self.root, 'pcd_*.ply')))
        self.annotation_files = [f.replace('.ply','_annotations.npy') for f in self.ply_files]
        

        # 並列処理でラベルを読み込み
        # print("Scanning dataset for classes...")
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     all_labels = pool.map(load_labels, self.annotation_files)
        
        # 結果を結合
        # all_labels = np.concatenate(all_labels)
        # self.unique_classes = np.unique(all_labels)
        # self.num_classes = len(self.unique_classes)
        self.unique_classes = np.array([0, 1, 2, 3, 4, 5])
        self.num_classes = len(self.unique_classes)
        # [0 1 2 3 4 5]
        print(f"Found {self.num_classes} unique classes: {self.unique_classes}")
    
    def __len__(self):
        return len(self.ply_files)

    # def voxelize_point_cloud(self, pcd, labels):
    #     # ボクセル化
    #     voxel_down_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
    #     voxel_points = np.asarray(voxel_down_pcd.points)
    #     # 最近傍法でラベルを割り当て
    #     tree = o3d.geometry.KDTreeFlann(pcd)
    #     voxel_labels = []
        
    #     for voxel_point in voxel_points:
    #         print(f"Searching for nearest neighbor for voxel point {voxel_point}")
    #         [k, idx, _] = tree.search_knn_vector_3d(voxel_point, 1)
    #         voxel_labels.append(labels[idx[0]])
        
        
    #     return voxel_points, np.array(voxel_labels)

    def voxelize_point_cloud(self, pcd, labels):
        # ボクセル化
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        voxel_points = np.asarray(voxel_down_pcd.points)
        
        # KDTreeを使用して最近傍法でラベルを割り当て
        tree = KDTree(np.asarray(pcd.points))
        
        # 並列処理でラベルを割り当て
        with mp.Pool(processes=mp.cpu_count()) as pool:
            voxel_labels = pool.map(partial(voxelize_point_cloud_worker, tree=tree, labels=labels), voxel_points)
        
        return voxel_points, np.array(voxel_labels)

    def __getitem__(self, idx):
        ply_file = self.ply_files[idx]
        anno_file = self.annotation_files[idx]

        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points, dtype=np.float32)
        labels = np.load(anno_file).astype(np.int64)
        
        # ボクセル化を適用
        # points, labels = self.voxelize_point_cloud(pcd, labels)
        
        # 必要な点数にリサンプリング
        M = points.shape[0]
        if M > self.num_points:
            choice = np.random.choice(M, self.num_points, replace=False)
        else:
            choice = np.random.choice(M, self.num_points, replace=True)
        
        points = points[choice, :]
        labels = labels[choice]

        if self.transform:
            points = self.transform(points)

        return points, labels


def main():
    # パラメータ設定
    num_points = 7500000  # より現実的な値に調整
    voxel_size = 0.1
    initial_lr = 0.001
    root_dir = '../dataset'  # Mac用の適切なパスを設定
    
    try:
        # デバイス設定
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {device}")

        # データセット初期化
        dataset = SegmentationDataset(root=root_dir, num_points=num_points, voxel_size=voxel_size)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # バッチサイズを小さくしてメモリ管理を改善
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

        model = PointNetSeg(num_points, dataset.num_classes).to(device)
        model = model.float()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        lrs = []

        for epoch in range(10):
            print(f"Epoch {epoch+1}/10")
            model.train()
            train_loss = 0.0
            for i in range(len(train_loader)):
                points, labels = next(iter(train_loader))
                try:
                    points = points.float().to(device)
                    labels = labels.long().to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(points)
                    outputs = outputs.permute(0, 2, 1).contiguous()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    print(f'Epoch [{epoch+1}/10], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                    print(f'Current Allocated Memory: {torch.cuda.memory_allocated(device)/1e9 if device.type == "cuda" else torch.mps.current_allocated_memory()/1e9:.2f} GB')
                    print(f'Current Cached Memory: {torch.cuda.memory_reserved(device)/1e9 if device.type == "cuda" else torch.mps.driver_allocated_memory()/1e9:.2f} GB')
                except RuntimeError as e:
                    print(f"Error during training: {e}")
                    if "out of memory" in str(e):
                        if device.type == 'cuda' and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif device.type == 'mps':
                            torch.mps.empty_cache()
                    continue

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # 検証フェーズ
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for points, labels in val_loader:
                    points = points.float().to(device)
                    labels = labels.long().to(device)
                    outputs = model(points)
                    outputs = outputs.permute(0, 2, 1).contiguous()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            print(f'Current Learning Rate: {current_lr:.6f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, 'best_model_checkpoint.pth')

            print(f'Epoch [{epoch+1}/10], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # グラフの保存
        plt.figure()
        plt.plot(range(1, 11), train_losses, label='Train Loss')
        plt.plot(range(1, 11), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig('loss_graph.png')

        plt.figure()
        plt.plot(range(1, 11), lrs, label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.title('Learning Rate Schedule')
        plt.savefig('lr_graph.png')

        # CSVファイルの保存
        with open('training_data.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Learning Rate'])
            for epoch in range(10):
                writer.writerow([epoch + 1, train_losses[epoch], val_losses[epoch], lrs[epoch]])

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    main()
