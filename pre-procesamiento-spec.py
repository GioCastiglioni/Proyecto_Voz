import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import skimage.io
import os
import h5py
import pandas
from tqdm import tqdm

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)

output_folder = "imgs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

datos = []
for root, dirs, files in os.walk("C:\\Users\\nicol\\OneDrive\\Semestre casi final\\Procesamiento de voz\\Proyecto\\AudioWAV"):
    for file in tqdm(files):
        if file[-4:] == ".wav":
            try:
                # load audio. Using example from librosa
                sr, y = wavfile.read(root+ "\\" +file, 'r')
                y = y.astype(np.float32)
                y = (y - np.min(y)) / (np.max(y) - np.min(y))
                out = os.path.join(output_folder, f'{file[:-4]}.png')

                # settings
                n_mels = 128  # number of bins in spectrogram. Height of image
                time_steps = 128  # number of time-steps. Width of image
                hop_length = int(len(y) / time_steps)  # adjust hop_length dynamically

                # extract a fixed length window
                start_sample = 0  # starting at the beginning
                length_samples = time_steps * hop_length
                window = y[start_sample:start_sample + length_samples]

                # convert to PNG
                spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
                # print('wrote file', out)
            except:
                print(f"Falla en archivo {file}")
        else:
            print(f"Falla en archivo {file}")
            pass