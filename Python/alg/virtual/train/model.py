import torch
import torch.nn as nn


class NonLinear(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(NonLinear, self).__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.main = nn.Sequential(
            nn.Linear(self.input_ch, self.output_ch),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.output_ch),
        )

    def forward(self, input_data):
        return self.main(input_data)


class MaxPool(nn.Module):
    def __init__(self, input_channels, num_kernel):
        super(MaxPool, self).__init__()
        self.num_channels = input_channels
        self.num_kernel = num_kernel
        self.main = nn.MaxPool1d(self.num_kernel)

    def forward(self, input_data):
        out = input_data.view(-1, self.num_channels, self.num_kernel)
        out = self.main(out)
        out = out.view(-1, self.num_channels)
        return out


class InputTNet(nn.Module):
    def __init__(self, num_points):
        super(InputTNet, self).__init__()
        self.num_points = num_points

        self.transT = nn.Sequential(
            NonLinear(3, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            # max pooling by number of points
            MaxPool(input_channels=1024, num_kernel=self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            # make 9 dimension feature for transformation matrix
            nn.Linear(256, 9),
        )

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        matrix = self.transT(input_data).view(-1, 3, 3)
        # transform whole point cloud
        out = torch.matmul(input_data.view(-1, self.num_points, 3), matrix)
        out = out.view(-1, 3)
        return out


class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNet, self).__init__()
        self.num_points = num_points

        self.featT = nn.Sequential(
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            # max pooling by number of points
            MaxPool(input_channels=1024, num_kernel=self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            # 64 x 64
            nn.Linear(256, 4096),
        )

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        matrix = self.featT(input_data).view(-1, 64, 64)
        # transform whole point cloud
        out = torch.matmul(input_data.view(-1, self.num_points, 64), matrix)
        out = out.view(-1, 64)
        return out


class PointNet(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.pointNet = nn.Sequential(
            InputTNet(self.num_points),
            NonLinear(3, 64),
            NonLinear(64, 64),
            FeatureTNet(self.num_points),
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            nn.Dropout(p=0.3),
            NonLinear(512, 256),
            nn.Dropout(p=0.3),
            # classification label
            NonLinear(256, self.num_labels),
        )

    def forward(self, input_data):
        return self.pointNet(input_data)
