import sys
sys.path.append('./src/')
import torch
from pathlib import Path
from src import models
from src import utils
import glob
import torchaudio
from adarsh_Targeted_Audio_Search.src.dataset2 import cnn14ModelConfig
import soundfile as sf
import librosa
import argparse
import numpy as np

cnn14_model_config = cnn14ModelConfig()
def get_feat_ref(path): 
        
  
    waveform, sr = torchaudio.load(path)
    audio_mono = torch.mean(waveform, dim=0, keepdim=True) # Need to change this function to extract features, use librosa and set the sample rate as 16000 and pad if needed or cut down

    # resample to 16khz
    resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=cnn14_model_config.sample_rate)
    audio_mono = resample_transform(audio_mono)
    
    tempData = torch.zeros([1, 160000])
    if audio_mono.numel() < 160000:
        tempData[:, :audio_mono.numel()] = audio_mono
    else:
        tempData = audio_mono[:, :160000]
    
    audio_mono=tempData
    
    mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate = cnn14_model_config.sample_rate, n_fft = cnn14_model_config.window_size,
                                                        win_length = cnn14_model_config.window_size, hop_length = cnn14_model_config.hop_size,
                                                        f_min = cnn14_model_config.fmin, f_max = cnn14_model_config.fmax,
                                                        n_mels = cnn14_model_config.mel_bins)(audio_mono)
    
    mfcc = torchaudio.transforms.MFCC(sample_rate=cnn14_model_config.sample_rate, 
                                        melkwargs = {"n_fft" : cnn14_model_config.window_size,
                                                    "win_length" : cnn14_model_config.window_size,
                                                    "hop_length" : cnn14_model_config.hop_size,
                                                    "n_mels" : cnn14_model_config.mel_bins,
                                                    "center" : True, "pad_mode" : cnn14_model_config.pad_mode})(audio_mono)
    
    new_feat = torch.cat([mel_specgram, mfcc], axis=1)
    new_feat = new_feat.permute(0, 2, 1).squeeze()
    return new_feat


parser = argparse.ArgumentParser()
parser.add_argument('-sr', type=int, default=16000) 
parser.add_argument('-winlen', default=40, type=float, help='FFT duration in ms')
parser.add_argument('-hoplen', default=20, type=float, help='hop duration in ms')
parser.add_argument('-n_mels', default=64, type=int) 
parser.add_argument('-audio_target_length', default=160000, type=int) # default to sr * 10 for 10 sec audio

ARGS = parser.parse_args()
EPS = np.spacing(1)

def get_feat_mix(fname):
    
    MEL_ARGS = {
                'n_mels': ARGS.n_mels,
                'n_fft': 1024,
                'hop_length': int(ARGS.sr * ARGS.hoplen / 1000),
                'win_length': int(ARGS.sr * ARGS.winlen / 1000)
             }
    y, sr = sf.read(fname, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=0)
    y = librosa.resample(y, orig_sr = sr, target_sr = ARGS.sr)

    if len(y) < ARGS.audio_target_length:
        
        missing = max(ARGS.audio_target_length - y.shape[0], 0)
        y = np.pad(y, missing, mode="constant", value=0.0)
    
    elif len(y) > ARGS.audio_target_length:
        y = y[:ARGS.audio_target_length]

    # compute melspec and return it
    lms_feature = np.log(librosa.feature.melspectrogram(y=y, **MEL_ARGS) + EPS).T
    return torch.tensor(lms_feature)


def evaluate(
            mix_mel: torch.Tensor,
            ref_mel: torch.Tensor,
            experiment_path: str,
            time_ratio=10. / 500,
            postprocessing='median',
            threshold=None,
            window_size=None,
            DEVICE = "cpu",
            **kwargs):
    
  
    config = torch.load(list(Path(f'{experiment_path}').glob("run_config*"))[0], map_location=DEVICE)
    config_parameters = dict(config, **kwargs)

    model_parameters = torch.load(glob.glob("{}/run_model*".format(experiment_path))[0],
                                    map_location=lambda storage, loc: storage)   

    # Load the model
    model = getattr(models, config_parameters['model'])(
        model_config=config_parameters,inputdim=64, outputdim=2, **config_parameters['model_args'])
    
    try:
        model.load_state_dict(model_parameters)
        print("Successfully loaded model")
    except:
        print("Unsuccessful in loading model")
        sys.exit(1)

    model = model.to(DEVICE).eval()
    with torch.no_grad():
        mix_mel = mix_mel.to(DEVICE)
        ref_mel = ref_mel.to(DEVICE)
        decision, decision_up, logit = model.forward(mix_mel, ref_mel)
        
        pred = decision_up.detach().cpu().numpy()
        pred = pred[:,:,0]

        filtered_pred = utils.median_filter(pred, window_size=window_size, threshold=threshold)
        
        decoded_pred = []
        time_predictions = []
        decoded_pred_ = utils.decode_with_timestamps("test_event",filtered_pred[0,:])
        if len(decoded_pred_) == 0: # neg deal
            print("Event not detected!")
            sys.exit(1)
        decoded_pred.append(decoded_pred_)


        
        for idx in range(len(decoded_pred)):
    
            label_prediction = decoded_pred[idx]
            for event_label, onset, offset in label_prediction:
                time_predictions.append({
                    'onset': onset * time_ratio,
                    'offset': offset * time_ratio,
                    'event_label': event_label})
            print("Predicted Timestamps", time_predictions)
        
    
if __name__ == "__main__":
    
    mix_path = "/home/ananth/datasets/URBAN-SED/audio/test/soundscape_test_bimodal1.wav"
    ref_path = "/home/ananth/datasets/UrbanSound8K/audio/fold1/102305-6-0-0.wav"

    mix_mel = get_feat_mix(mix_path).unsqueeze(0)
    ref_mel = get_feat_ref(ref_path).unsqueeze(0)

    # mix_mel = torch.rand(1,501,64) # mixture mel specgram 
    # ref_mel = torch.rand(1,1001,104) # reference (mel+mfcc) features
    
    experiment_path = "./experiments/Join_fusion/2024-08-10_17-15-08_03c282e8570e11efaca83cecefb1a652/" 
    time_ratio = 10.0/500

    config_file = "./runconfigs/target_sed_join_train.yaml"
    config_parameters = utils.parse_config_or_kwargs(config_file)

    threshold = 0.4 # default threshold
    print("Threshold:",threshold)
    print("Threshold is adjustable, with a default setting of 0.4, as it performs best on our test data. You can modify it based on the desired sensitivity")
    
    postprocessing = config_parameters.get('postprocessing', 'double')  
    window_size = 1
    
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu" 
    print("DEVICE:",DEVICE)
    
    evaluate(mix_mel,
            ref_mel,
            experiment_path,
            time_ratio=time_ratio,
            postprocessing=postprocessing,
            threshold=threshold,
            window_size=window_size,
            DEVICE = DEVICE)