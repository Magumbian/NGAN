# Copyright (C) 2023 Katsuya Iida. All rights reserved.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
from ssr_eval.lowpass import *
import fnmatch
from typing import Tuple, List, Optional
import torch
torch.manual_seed(42)
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.nn import init
from encodec.msstftd import MultiScaleSTFTDiscriminator
from audio_to_mel import Audio2Mel
import scipy.signal
from scipy.interpolate import interp1d, CubicSpline
device = torch.device("cuda:0")

try:
    import pytorch_lightning as pl
except ImportError:
    class pl:
        class LightningModule:
            pass

        class Callback:
            pass
from itertools import chain
import random

l1Loss = torch.nn.L1Loss(reduction='mean')
l2Loss = torch.nn.MSELoss(reduction='mean')

def cubic_upsampler(audio_signal, input_rate, target_rate):
    original_duration = audio_signal.size()[-1] / input_rate
    original_time = np.linspace(0, original_duration, audio_signal.size()[-1])
    target_time = np.linspace(0, original_duration, int(original_duration * target_rate))
    cubic_spline = CubicSpline(original_time, audio_signal, axis=-1)
    upsampled_audio = cubic_spline(target_time)
    return upsampled_audio

class ResNet1d(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size: int = 3,
        padding: str = 'same',
        dilation: int = 1
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self._padding_size = (kernel_size // 2) * dilation
        self.conv0 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation)
        self.bn0 = nn.BatchNorm1d(
            n_channels
        )
        self.conv1 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=1)
        self.bn1 = nn.BatchNorm1d(
            n_channels
        )

    def forward(self, input):
        y = input
        x = self.conv0(input)
        #x = self.bn0(x)
        x = F.elu(x)
        x = self.conv1(x)
        #x = self.bn1(x)
        if self.padding == 'valid':
            y = y[:, :, self._padding_size:-self._padding_size]
        x += y
        x = F.elu(x)
        return x


class ResNet2d(nn.Module):
    def __init__(
        self,
        n_channels: int,
        factor: int,
        stride: Tuple[int, int]
    ) -> None:
        # https://arxiv.org/pdf/2005.00341.pdf
        super().__init__()
        self.conv0 = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=(3, 3),
            padding='same')
        self.bn0 = nn.BatchNorm2d(
            n_channels
        )
        self.conv1 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=(stride[0] + 2, stride[1] + 2),
            stride=stride)
        self.bn1 = nn.BatchNorm2d(
            factor * n_channels
        )
        self.conv2 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=1,
            stride=stride)
        self.bn2 = nn.BatchNorm2d(
            factor * n_channels
        )
        self.pad = nn.ReflectionPad2d([
            (stride[1] + 1) // 2,
            (stride[1] + 2) // 2,
            (stride[0] + 1) // 2,
            (stride[0] + 2) // 2,
        ])
        self.activation = nn.LeakyReLU(0.3)

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = self.activation(x)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)

        # shortcut
        y = self.conv2(input)
        y = self.bn2(y)
        x += y
        x = self.activation(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
            nn.Conv1d(
                n_channels // 2, n_channels,
                kernel_size=2 * stride,
                padding=stride// 2,
                stride=stride),
            nn.ELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(
                n_channels, n_channels // 2,
                kernel_size=2 * stride,
                padding=stride// 2,
                stride=stride),
            nn.ELU(),
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class SuperResolver(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']
        self.n_channels = n_channels
        self.resolve = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            EncoderBlock(2 * n_channels, padding=padding, stride=6),
            EncoderBlock(4 * n_channels, padding=padding, stride=4),
            EncoderBlock(8 * n_channels, padding=padding, stride=2),
            nn.Conv1d(8 * n_channels, 8 * n_channels, kernel_size=3, padding=padding),
            nn.Conv1d(8 * n_channels, 8 * n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            DecoderBlock(8 * n_channels, padding=padding, stride=6),
            DecoderBlock(4 * n_channels, padding=padding, stride=4),
            DecoderBlock(2 * n_channels, padding=padding, stride=2),
            nn.Conv1d(n_channels, 1, kernel_size=7, padding=padding),
            nn.Tanh(),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.resolve(input[1][:, None, :])
        return x

class ReconstructionLoss(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    but uses STFT instead of mel-spectrogram
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = 0
        input = input.to(torch.float32)
        target = target.to(torch.float32)
        for i in range(6, 12):
            s = 2 ** i
            alpha = (s / 2) ** 0.5
            # We use STFT instead of 64-bin mel-spectrogram as n_fft=64 is too small
            # for 64 bins.
            x = torch.stft(input, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            x = torch.abs(x)
            y = torch.stft(target, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            y = torch.abs(y)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, :, :y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, :, :x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        return loss / (12 - 6)


def _get_flist(folder):
    wav_files = []
    for root, dirs, files in os.walk(folder):
        for filename in files: 
            if filename.endswith(('.wav', '.flac', '.mp4')):
                wav_files.append(os.path.join(root, filename))
    return wav_files
    
class NGAN(pl.LightningModule):
    def __init__(
        self,
        n_channels: int = 8,
        padding: str = 'check',
        batch_size: int = 32,
        target_sample_rate: int = 48000,
        segment_length: int = 24000,
        train_data: str = None,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        dataset: str = 'librispeech'
    ) -> None:
        # https://arxiv.org/pdf/2009.02095.pdf
        # 2. Method
        # SEANet uses Adam with lr=1e-4, beta1=0.5, beta2=0.9
        # batch_size=16
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.superResolver = SuperResolver(n_channels, padding)
        self.rec_loss = ReconstructionLoss()
        self.msstft_discriminator = MultiScaleSTFTDiscriminator(filters=32,
                                             hop_lengths=[256, 512, 128],
                                             win_lengths=[1024, 2048, 512],
                                             n_ffts=[1024, 2048, 512])
        

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_g = torch.optim.Adam(
            chain(
                self.superResolver.parameters()
            ),
            lr=lr, betas=(b1, b2))
        optimizer_d = torch.optim.Adam(
            chain(
                self.msstft_discriminator.parameters()
            ),
            lr=lr, betas=(b1, b2))
        return [optimizer_g, optimizer_d], []

    def forward(self, input):
        x = self.superResolver(input)
        return x

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()       

        # train generator
        self.toggle_optimizer(optimizer_g)
        output = self.forward(batch)
        # output: [batch, channel, sequence]
        
        #multi-scale stft generator loss
        relu = torch.nn.ReLU()
        l_t = 0
        l_f = 0
        l_g = 0
        l_feat = 0
        input_orig = batch[0][:, None, :]
        logits_real, fmap_real = self.msstft_discriminator(input_orig)
        logits_fake, fmap_fake = self.msstft_discriminator(output.detach())
        for tt1 in range(len(fmap_real)): 
            l_g = l_g + torch.mean(relu(1 - logits_fake[tt1])) / len(logits_fake)
            for tt2 in range(len(fmap_real[tt1])):
                l_feat = l_feat + l1Loss(fmap_real[tt1][tt2], fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2]))
        KL_scale = len(fmap_real)*len(fmap_real[0])
        K_scale = len(fmap_real)
        
        #time domain loss, output_wav is the output of the generator
        l_t = l1Loss(input_orig, output) 
            
        g_rec_loss = self.rec_loss(output[:, 0, :], input_orig[:, 0, :])
        
        g_loss = 3*l_g/K_scale + 3*l_feat/KL_scale + (l_t / 10) + g_rec_loss 

        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        relu = torch.nn.ReLU()
        logits_real, _ = self.msstft_discriminator(input_orig)
        logits_fake, _ = self.msstft_discriminator(output.detach())
        lossd = 0
        for tt1 in range(len(logits_real)):
            lossd = lossd + torch.mean(relu(1-logits_real[tt1])) + torch.mean(relu(1+logits_fake[tt1]))
        d_loss = lossd / len(logits_real)
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def train_dataloader(self):
        return self._make_dataloader(True)

    def _make_dataloader(self, train: bool):
        import torchaudio

        def collate(examples):
            x_list = [example[0] for example in examples]
            y_list = [example[1] for example in examples]
            x_stack = torch.stack(x_list)
            y_stack = torch.stack(y_list)
            return [x_stack, y_stack]
        
        class VoiceDataset(torch.utils.data.Dataset):
            def __init__(self, root: str = self.hparams['train_data'], segment_length=self.hparams['segment_length'], target_sample_rate=self.hparams['target_sample_rate']) -> None:
                self._flist = _get_flist(root)
                self.segment_length = segment_length
                self.target_sample_rate=target_sample_rate

                       
            def __getitem__(self, n: int):
                x, sr = torchaudio.load(self._flist[n])
                resample_transform_x = T.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
                if sr > 48000:
                    x = resample_transform_x(x)
                if x.size()[0] == 2:
                    x = x[random.randint(0, 1)].unsqueeze(dim=0)                
                assert x.shape[0] == 1
                x = torch.squeeze(x)
                assert x.dim() == 1
                if x.shape[0] < self.segment_length:
                    x = F.pad(x, [0, self.segment_length - x.shape[0]], "constant")
                pos = random.randint(0, x.shape[0] - self.segment_length)
                x = x[pos:pos + self.segment_length]
                filters = ["butter", "cheby1", "ellip", "bessel"]
                freqs = [8000, 16000, 24000]
                freq = random.choice(freqs)
                cutoff, fs, order = freq*0.5, self.target_sample_rate, 8
                z = lowpass_filter(x, cutoff, fs, order, ftype=random.choice(filters))
                z = librosa.resample(z, orig_sr=self.target_sample_rate, target_sr=freq)
             
                v = torch.tensor(z.copy(), dtype=torch.float32)
                v_ = cubic_upsampler(v, freq, self.target_sample_rate)                
                y = torch.tensor(v_, dtype=torch.float32)             
                vals = [x, y]           
                return vals

            def __len__(self) -> int:
                return len(self._flist)
        
        ds = VoiceDataset()

        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams['batch_size'], shuffle=True,
            collate_fn=collate)
        return loader
        
        


def train():
    model = NGAN(
        batch_size = 32,
        padding = 'same',
        dataset = 'librispeech-dev')
    #checkpoint = torch.load('versatility/small/versatile-v5.ckpt')['state_dict']     
    #model.load_state_dict(checkpoint)
    #torch.save(model.superResolver.state_dict(), 'generators/n1x.ckpt')
    trainer = pl.Trainer(
        max_epochs=200,
        precision='16-mixed',
        logger=pl.loggers.CSVLogger("."),
        # logger=pl.loggers.TensorBoardLogger("lightning_logs", name="soundstream"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True, every_n_train_steps=50000, filename='versatile', dirpath='./versatility/small/')
        ],
    )
    trainer.fit(
        model,
    )

    return model


if __name__ == "__main__":
    train()    

    
       
