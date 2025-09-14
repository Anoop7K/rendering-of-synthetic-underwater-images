import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()

        def down_block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_f, out_f, dropout=0.0):
            layers = [
                nn.ConvTranspose2d(in_f, out_f, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.down1 = down_block(in_channels, features, normalize=False)
        self.down2 = down_block(features, features*2)
        self.down3 = down_block(features*2, features*4)
        self.down4 = down_block(features*4, features*8)
        self.down5 = down_block(features*8, features*8)

        self.up1 = up_block(features*8, features*8)
        self.up2 = up_block(features*16, features*4)
        self.up3 = up_block(features*8, features*2)
        self.up4 = up_block(features*4, features)
        self.final = nn.ConvTranspose2d(features*2, out_channels, 4, stride=2, padding=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], 1))
        u3 = self.up3(torch.cat([u2, d3], 1))
        u4 = self.up4(torch.cat([u3, d2], 1))
        return torch.tanh(self.final(torch.cat([u4, d1], 1)))
