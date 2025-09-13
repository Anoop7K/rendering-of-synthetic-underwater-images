import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=9, features=64):
        super(PatchGANDiscriminator, self).__init__()

        def block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, features, normalize=False),
            block(features, features*2),
            block(features*2, features*4),
            nn.Conv2d(features*4, 1, 4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
