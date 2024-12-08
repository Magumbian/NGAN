# import git
# git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
# import sys
# sys.path.append(git_root)
import os
import librosa
import torch
import numpy as np
from scipy.signal import resample_poly
from subjective_evals.plc_mos import PLCMOSEstimator
from skimage.metrics import structural_similarity as ssim
from ssr_eval.utils import *
from subjective_evals.test_dnsmos import ComputeScore
from visqol import visqol_lib_py 
from visqol.pb2 import visqol_config_pb2 
from visqol.pb2 import similarity_result_pb2
from pesq import pesq

EPS = 1e-12


class AudioMetrics:
    def __init__(self, rate):
        self.rate = rate
        self.hop_length = int(rate / 100)
        self.n_fft = int(2048 / (48000 / rate))

    def read(self, est, target):
        est, _ = librosa.load(est, sr=self.rate, mono=True)
        target, _ = librosa.load(target, sr=self.rate, mono=True)
        return est, target

    def wav_to_spectrogram(self, wav):
        f = np.abs(librosa.stft(wav, hop_length=self.hop_length, n_fft=self.n_fft))
        f = np.transpose(f, (1, 0))
        f = torch.tensor(f[None, None, ...])
        return f

    def center_crop(self, x, y):
        dim = 2
        if x.size(dim) == y.size(dim):
            return x, y
        elif x.size(dim) > y.size(dim):
            offset = x.size(dim) - y.size(dim)
            start = offset // 2
            end = offset - start
            x = x[:, :, start:-end, :]
        elif x.size(dim) < y.size(dim):
            offset = y.size(dim) - x.size(dim)
            start = offset // 2
            end = offset - start
            y = y[:, :, start:-end, :]
        assert (
            offset < 10
        ), "Error: the offset %s is too large, check the code please" % (offset)
        return x, y

    def evaluation(self, est, target, file):
        """evaluate between two audio
        Args:
            est (str or np.array): _description_
            target (str or np.array): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        # import time; start = time.time()
        if type(est) != type(target):
            raise ValueError(
                "The input value should either both be numpy array or strings"
            )
        if type(est) == type(""):
            est_wav, target_wav = self.read(est, target)
        else:
            assert len(list(est.shape)) == 1 and len(list(target.shape)) == 1, (
                "The input numpy array shape should be [samples,]. Got input shape %s and %s. "
                % (est.shape, target.shape)
            )
            est_wav, target_wav = est, target

        # target_spec_path = os.path.join(os.path.dirname(file), os.path.splitext(os.path.basename(file))[0]+"_proc_%s.pt" % (self.rate))
        # if(os.path.exists(target_spec_path)):
        #     target_sp = torch.load(target_spec_path)
        # else:

        assert (
            abs(target_wav.shape[0] - est_wav.shape[0]) < 100
        ), "Error: Shape mismatch between target and estimation %s and %s" % (
            str(target_wav.shape),
            str(est_wav.shape),
        )

        min_len = min(target_wav.shape[0], est_wav.shape[0])
        target_wav, est_wav = target_wav[:min_len], est_wav[:min_len]

        target_sp = self.wav_to_spectrogram(target_wav)
        est_sp = self.wav_to_spectrogram(est_wav)

        result = {}

        # frequency domain
        result["lsd"] = self.lsd(est_sp.clone(), target_sp.clone())
        result["log_sispec"] = self.sispec(
            to_log(est_sp.clone()), to_log(target_sp.clone())
        )
        est_16k = resample_poly(est, 16000, 48000)
        target_16k = resample_poly(target, 16000, 48000)
        result["sispec"] = self.sispec(est_sp.clone(), target_sp.clone())
        result["ssim"] = self.ssim(est_sp.clone(), target_sp.clone())
        result["plcmos"] = self.plcmos(est_16k)
        result["utmos22"] = self.utmos22(est)
        result["dnsmos"] = self.dnsmos(est)
        result["visqol"] = self.visqol(est_16k, target_16k)
        result["pesq"] = (pesq(16000, target_16k, est_16k, 'wb'))


        for key in result:
            result[key] = float(result[key])
        return result

    def lsd(self, est, target):
        lsd = torch.log10(target**2 / ((est + EPS) ** 2) + EPS) ** 2
        lsd = torch.mean(torch.mean(lsd, dim=3) ** 0.5, dim=2)
        return lsd[..., None, None]

    def plcmos(self, est):
        predictor = PLCMOSEstimator()
        #est_16k = resample_poly(est, 16000, 48000)
        return predictor.run(est, 16000)

    def utmos22(self, est):
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        return predictor(torch.from_numpy(est).unsqueeze(0), 48000)

    def dnsmos(self, est):
        p808_model_path = os.path.join('/home/mark/.local/lib/python3.11/site-packages/ssr_eval/DNSMOS', 'model_v8.onnx')
        primary_model_path = os.path.join('/home/mark/.local/lib/python3.11/site-packages/ssr_eval/DNSMOS', 'sig_bak_ovr.onnx') 
        predictor = ComputeScore(primary_model_path, p808_model_path)
        return predictor.mos(est, False)['P808_MOS']
    
    def visqol(self, est, target):
        config = visqol_config_pb2.VisqolConfig()
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True     
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
        config.options.svr_model_path = os.path.join(     
                os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)  
        
        api = visqol_lib_py.VisqolApi() 
        api.Create(config)   
        #est_16k = resample_poly(est, 16000, 48000)
        #target_16k = resample_poly(target, 16000, 48000)
        similarity_result = api.Measure(target.astype(np.float64), est.astype(np.float64))
        return similarity_result.moslqo




    def sispec(self, est, target):
        # in log scale
        output, target = energy_unify(est, target)
        noise = output - target
        sp_loss = 10 * torch.log10(
            (pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
        )
        return torch.sum(sp_loss) / sp_loss.size()[0]

    def ssim(self, est, target):
        if "cuda" in str(target.device):
            target, output = target.detach().cpu().numpy(), est.detach().cpu().numpy()
        else:
            target, output = target.numpy(), est.numpy()
        res = np.zeros([output.shape[0], output.shape[1]])
        data_range = max(np.max(target), np.max(output)) - min(np.min(target), np.min(output))
        for bs in range(output.shape[0]):
            for c in range(output.shape[1]):
                res[bs, c] = ssim(output[bs, c, ...], target[bs, c, ...], win_size=7, data_range = data_range)
        return torch.tensor(res)[..., None, None]

if __name__ == "__main__":
    import numpy as np

    au = AudioMetrics(rate=44100)
    # path1 = "old/out.wav"
    path1 = "eeeeee.wav"
    # path2 = "old/target.wav"
    path2 = "targete.wav"
    result = au.evaluation(path2, path1, path1)
    print(result)
