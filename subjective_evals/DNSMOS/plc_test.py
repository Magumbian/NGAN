import soundfile as sf
from plc_mos import PLCMOSEstimator 
plcmos = PLCMOSEstimator()  
data, sr = sf.read("example_wavs/clean.wav")
print(data.shape)
mos = plcmos.run(data, sr)
print(mos)
