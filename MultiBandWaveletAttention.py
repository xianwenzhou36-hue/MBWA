import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse


class DWT1D(nn.Module):
    def __init__(self, wavelet: str = 'db6', level: int = 5):
        super().__init__()
        self.transform = DWT1DForward(J=level, wave=wavelet, mode='symmetric')

    def forward(self, x):
        yl, yh = self.transform(x)
        subbands = list(yh) + [yl]
        return subbands


class IWT1D(nn.Module):
    def __init__(self, wavelet: str = 'db6'):
        super().__init__()
        self.inverse = DWT1DInverse(wave=wavelet, mode='symmetric')

    def forward(self, subbands):
        yl = subbands[-1]
        yh = subbands[:-1]
        return self.inverse((yl, yh))


class ChannelAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )

    def forward(self, x):
        avg = x.mean(dim=2)
        max_ = x.max(dim=2)[0]
        att = torch.sigmoid(self.mlp(avg) + self.mlp(max_))
        return x * att.unsqueeze(-1)


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_ = x.max(dim=1, keepdim=True)[0]
        att = torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * att


class CBAM1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention1D(channels, reduction)
        self.spatial = SpatialAttention1D()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

class IntraBandAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention1D(channels, reduction)
        self.temporal = SpatialAttention1D()

    def forward(self, x):
        x = self.channel(x)
        x = self.temporal(x)
        return x


class InterBandAttention(nn.Module):
    def __init__(self, num_bands: int, reduction: int = 2):
        super().__init__()
        hidden = max(num_bands // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(num_bands, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_bands, bias=False)
        )

    def forward(self, subbands):
        statistics = [sb.mean(dim=(1, 2)) for sb in subbands]
        statistics = torch.stack(statistics, dim=1)
        weights = F.softmax(self.fc(statistics), dim=1)
        outputs = []
        for i, sb in enumerate(subbands):
            w = weights[:, i].view(-1, 1, 1)
            outputs.append(sb * w)
        return outputs


class WaveletDomainAttention(nn.Module):
    def __init__(self, channels: int, wavelet: str = 'db6', level: int = 5, reduction: int = 16):
        super().__init__()
        self.dwt = DWT1D(wavelet, level)
        self.iwt = IWT1D(wavelet)
        self.intra = IntraBandAttention(channels, reduction)
        self.inter = InterBandAttention(level + 1, reduction=2)

    def forward(self, x):
        subbands = self.dwt(x)
        subbands = [self.intra(sb) for sb in subbands]
        subbands = self.inter(subbands)
        out = self.iwt(subbands)
        return out


class MultiBandWaveletAttention(nn.Module):
    def __init__(self, channels: int, wavelet: str = 'db6', level: int = 5, reduction: int = 16):
        super().__init__()
        self.cbam = CBAM1D(channels, reduction)
        self.wda = WaveletDomainAttention(channels, wavelet, level, reduction)
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        cbam_out = self.cbam(x)
        wda_out = self.wda(x)
        out = torch.cat([cbam_out, wda_out, x], dim=1)
        out = self.fusion(out)
        return out


# test case
if __name__ == "__main__":
    torch.manual_seed(0)

    x1 = torch.randn(4, 64, 250)
    mbwa1 = MultiBandWaveletAttention(channels=64, wavelet='db6', level=5)
    y1 = mbwa1(x1)
    print("Stem:   ", x1.shape, "->", y1.shape)

    x2 = torch.randn(4, 128, 64)
    mbwa2 = MultiBandWaveletAttention(channels=128, wavelet='db6', level=3)
    y2 = mbwa2(x2)
    print("Middle: ", x2.shape, "->", y2.shape)

    x3 = torch.randn(4, 256, 20)
    mbwa3 = MultiBandWaveletAttention(channels=256, wavelet='sym4', level=2)
    y3 = mbwa3(x3)
    print("Final:  ", x3.shape, "->", y3.shape)

    params = sum(p.numel() for p in mbwa1.parameters() if p.requires_grad)
    print("Trainable parameters:", params)