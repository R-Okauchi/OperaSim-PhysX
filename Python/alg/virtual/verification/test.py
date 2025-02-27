import torch
from Model import PointNet
from sampler import PcdDataset

batch_size = 64
num_points = 1024
num_labels = 1
weight_path = "pointnet_seg_large_weights.pth"  # 学習済みモデルの重みファイルパス


def test():
    # GPU利用可能ならGPUを使う
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの読み込み
    pointnet = PointNet(num_points, num_labels).to(device)
    pointnet.load_state_dict(torch.load(weight_path, map_location=device))
    pointnet.eval()

    # テスト用データセット・データローダ
    # ここでは4件のデータをテスト用に用意 (for_test=True)
    dataset = PcdDataset(data_num=4, npoints=num_points, for_test=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for input_data, labels in dataloader:
            # (B * N, 3) 形式でモデルに渡す処理だが、
            # モデル次第で (B, N, 3) のまま渡せるように修正してもよい。
            input_data = input_data.view(-1, 3).to(device)
            labels = labels.view(-1, 1).to(device)

            pred = pointnet(input_data.float())
            pred = torch.sigmoid(pred)

            # 0.5を閾値として2値化
            pred_binary = (pred > 0.5).float()

            # 予測とラベルを比較
            correct_count += (pred_binary == labels).sum().item()
            total_count += labels.size(0)

            # サンプルごとの結果を表示 (任意)
            for predicted_val, actual_val in zip(pred_binary, labels):
                print(
                    f'Predicted: "{int(predicted_val.item())}", Actual: "{int(actual_val.item())}"',
                    "correct!"
                    if predicted_val.item() == actual_val.item()
                    else "incorrect...",
                )

    # 全データに対する精度表示
    accuracy = correct_count / total_count
    print(f"Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    test()
