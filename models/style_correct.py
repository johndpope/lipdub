from models import models_utils
from custom_types import *


class SkipCov(nn.Module):

    def forward(self, x: T) -> T:
        x = self.net(x) + x
        return x

    def __init__(self):
        super(SkipCov, self).__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(128),
                                 nn.ReLU(True),
                                 nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(True),
                                 nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
                                 )


class StyleCorrectNet(nn.Module):

    def forward(self, x: T) -> T:
        return self.net(x)

    def __init__(self, num_skips: int):
        super(StyleCorrectNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 128, (3, 3), (1, 1), (1, 1)),
                                 *[SkipCov() for _ in range(num_skips)],
                                 nn.Conv2d(128, 3, (3, 3), (1, 1), (1, 1))
                                 )


class StyleCorrect(models_utils.Model):

    def forward256(self, x):
        return self.net256(x)

    def forward(self, x_256: T, x_512: T) -> TS:
        return self.net256(x_256), self.net512(x_512)

    def __init__(self, opt):
        super(StyleCorrect, self).__init__()
        self.net256 = StyleCorrectNet(5)
        self.net512 = StyleCorrectNet(1)
