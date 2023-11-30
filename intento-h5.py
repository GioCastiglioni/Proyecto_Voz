import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import skimage.io
import os
import h5py
from tqdm import tqdm

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, group, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9)  # add a small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy

    # save as PNG and store as a string in the HDF5 file
    img_path = f'{file[:-4]}.png'
    skimage.io.imsave(img_path, img)
    group.create_dataset(name=f'{file[:-4]}', data=img_path, dtype=h5py.string_dtype())

# Create an HDF5 file
hdf5_file = h5py.File('spects.h5', 'w')

datos = []
for root, dirs, files in os.walk("C:\\Users\\nicol\\OneDrive\\Semestre casi final\\Procesamiento de voz\\Proyecto\\AudioWAV"):
    for file in tqdm(files):
        if file[-4:] == ".wav":
            try:
                # load audio. Using example from librosa
                sr, y = wavfile.read(os.path.join(root, file), 'r')
                y = y.astype(np.float32)
                y = (y - np.min(y)) / (np.max(y) - np.min(y))

                # settings
                n_mels = 128  # number of bins in spectrogram. Height of image
                time_steps = 128  # number of time-steps. Width of image
                hop_length = int(len(y) / time_steps)  # adjust hop_length dynamically

                # extract a fixed length window
                start_sample = 0  # starting at the beginning
                length_samples = time_steps * hop_length
                window = y[start_sample:start_sample + length_samples]

                # create a group for each spectrogram in the HDF5 file
                group = hdf5_file.create_group(file[:-4])
                spectrogram_image(window, sr=sr, group=group, hop_length=hop_length, n_mels=n_mels)
            except:
                print(f"Falla en archivo {file}")
        else:
            print(f"Falla en archivo {file}")
            pass

# Close the HDF5 file
hdf5_file.close()