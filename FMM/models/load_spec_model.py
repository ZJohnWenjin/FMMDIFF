import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2, 1), 
            nn.InstanceNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


def Normalize(in_channels):
    return nn.BatchNorm2d(in_channels)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        block = [
            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            Normalize(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            Normalize(n_channels),
            nn.ReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, num_down, base_ch=64):
        super().__init__()
        assert num_down >= 1, "num_down must be >= 1"


        chs = [base_ch * (2 ** i) for i in range(num_down)]
        self.out_channels_per_level = chs 

        blocks = []
        for i in range(num_down):
            in_ch = in_channels if i == 0 else chs[i - 1]
            out_ch = chs[i]
            blocks.append(nn.Sequential(
                Down(in_ch, out_ch),
                ResidualBlock(out_ch),
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x) 
        return feats  


class Decoder(nn.Module):
    def __init__(self, chs, out_channels=1):

        super().__init__()
        assert len(chs) >= 1
        self.num_down = len(chs)

        up_blocks = []

        for i in range(self.num_down - 1, 0, -1):
            up_blocks.append(nn.Sequential(
                Upsample(in_channels=chs[i], out_channels=chs[i - 1]),
                ResidualBlock(chs[i - 1]),
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final_upsample = Upsample(in_channels=chs[0], out_channels=out_channels)
        self.out_activation = nn.Sigmoid()

    def forward(self, feats):
        """
        feats: 来自 Encoder 的特征列表 [x1, x2, ..., xN]
        """
        assert isinstance(feats, (list, tuple)) and len(feats) == self.num_down
        y = feats[-1] 

        for k, i in enumerate(range(self.num_down - 1, 0, -1)):
            y = self.up_blocks[k](y)
            y = y + feats[i - 1]  

        y = self.final_upsample(y)  
        return self.out_activation(y)


class FMM_model(nn.Module):
    def __init__(self, in_channel, num_down):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channel, num_down=num_down, base_ch=64)
        self.decoder = Decoder(chs=self.encoder.out_channels_per_level, out_channels=1)

    def forward(self, x):
        feats = self.encoder(x)         
        out = self.decoder(feats)       
        return out


