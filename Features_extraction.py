import scipy.io.wavfile as wavfile
import numpy as np
import os
import h5py
import pandas
from tqdm import tqdm


def toeplitz(c):
    '''
    Devuelve una matriz toeplitz armada con el vector c
    '''
    n = len(c)
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j >= i:
                T[i, j] = c[j - i]
            else:
                T[i, j] = c[i - j]
    return T

def LPC(audio):
    ventanas = np.array([audio[i*len(audio)//19:(i+1)*len(audio)//19] for i in range(19)])
    
    # Se define el orden del análisis LPC (p)
    p = 13
    
    coeffs = np.zeros((19,p))
    
    # Se recorren las ventanas
    for k in range(len(ventanas)):
        ventana = ventanas[k] # Se trabaja con una ventana
        autocorrelation_values = np.correlate(ventana, ventana, mode='full')  # Se calcula la matriz de correlación completa de la ventana
        # Se comienza a construir la matriz R de dimensión pxp rescatando los R(0) hasta R(p)
        vector_R = np.zeros(p)
        R = np.zeros(p)
        for i in range(p):
            R[i] = autocorrelation_values[len(ventana) + i - 1]
            vector_R[i] = autocorrelation_values[len(ventana) + i]
    
        # Se construye finalmente la matriz de pxp a partir de los valores de autocorrelación y se calcula su inversa
        inversa = np.linalg.inv(toeplitz(R))
    
        # Se resuelve el sistema lineal para despejar los coeficientes
        coeffs[k] = np.matmul(inversa, vector_R)
        
    return coeffs
        

datos = []
for root, dirs, files in os.walk("C:\\Users\\gcast\\OneDrive\\Documentos\\Universidad\\Voz\\Proyecto\\CREMA-D\\AudioWAV"):
    for file in tqdm(files):
        if file[-4:] == ".wav":
            try:
                _, audio = wavfile.read(root+ "\\" +file)
                audio = audio[:len(audio) - len(audio)%19].astype(np.float32)
                lpc = LPC(audio).astype(np.float32)
                
                name = file[:-4]
                label = file.split('_')[2]
                
                with h5py.File("C:\\Users\\gcast\\OneDrive\\Documentos\\Universidad\\Voz\\Proyecto\\AudiosLPC.h5", 'a') as hf:
                    hf.create_dataset(name, data=lpc)
                    
                with h5py.File("C:\\Users\\gcast\\OneDrive\\Documentos\\Universidad\\Voz\\Proyecto\\AudiosWAV.h5", 'a') as hf:
                    hf.create_dataset(name, data=audio)
                    
                datos.append({'name': name,
                              'label': label})
            except:
                pass
                
pandas.DataFrame(datos).to_csv('C:\\Users\\gcast\\OneDrive\\Documentos\\Universidad\\Voz\\Proyecto\\Audios.csv', index=False)