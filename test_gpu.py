import torch
import time
import librosa
import numpy as np
from ssr_eval.eval import SSR_Eval_Helper, BasicTestee

from model import NGAN
#from load_balanced_model import NGAN

import scipy.signal 
from scipy.interpolate import interp1d, CubicSpline

class MyTestee(BasicTestee):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.model = NGAN(         
                batch_size=1,         
                target_sample_rate=48000,         
                segment_length=24000,         
                padding='same').to(device)
        #self.checkpoint = torch.load('generators/w4.ckpt') #To load load balanced model 
        self.checkpoint = torch.load('generators/n8.ckpt') #For best performing 8 channel model

    
    def cubic_upsampler(self, audio_signal, input_rate, target_rate):  
        original_duration = audio_signal.size / input_rate     
        original_time = np.linspace(0, original_duration, audio_signal.size)     
        target_time = np.linspace(0, original_duration, int(original_duration * target_rate))     
        cubic_spline = CubicSpline(original_time, audio_signal, axis=-1)     
        upsampled_audio = cubic_spline(target_time)     
        return upsampled_audio
    def infer(self, x):
        x = self.cubic_upsampler(x, 8000, 48000) #for 8KHz input
        x = torch.tensor(x.copy(), dtype=torch.float32).to(self.device)
        x = x.unsqueeze(dim=0)
        j = [x, x]
        self.model.superResolver.load_state_dict(self.checkpoint)
        with torch.no_grad():
            y = self.model.superResolver(j).squeeze(dim=0)
        out = self.tensor2numpy(y)[0]
        return out
    
if __name__ == "__main__":

    if(torch.cuda.is_available()): 
        device = "cuda"     
    else: device="cpu"

    testee = MyTestee(device)
    #"butter", "cheby1", "ellip", "bessel"
    helper = SSR_Eval_Helper(testee, 
                      test_name="8x48_performance", 
                      test_data_root="test_data", 
                      input_sr=8000,
                      output_sr=48000,
                      evaluation_sr=48000,
                      setting_lowpass_filtering = {  
                          "filter":["cheby", "bessel", "ellip", "butter"],
                          "cutoff_freq": [4000],
                          "filter_order": [8] },
                      save_processed_result=False,
    )
    
    helper.evaluate(limit_test_nums=-1, limit_test_speaker=-1)
