
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import librosa
from smFileRepresentation import smFile
from scipy.ndimage import zoom

def convertToFrequencyDomain(filename, sm:smFile):
    audio, samples_per_second = librosa.load(filename, sr=None)
    stft = librosa.stft(audio)

    spectrogram = np.abs(stft)

    samples_to_spec_samples = spectrogram.shape[1] / audio.shape[0]
    print(spectrogram_scale_factor)

    offset_seconds = 0 if sm is None else sm.offset
    offset_samples = offset_seconds * samples_per_second
    
    if offset_samples > 0:
        zero_array = np.zeros((spectrogram.shape[0], int(round(offset_samples * samples_to_spec_samples))))

    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return spectrogram_db, sample_rate

def getMeasure(spectrogram, total_measures, measure_number, dim=None):
    width = int(spectrogram.shape[1] / total_measures)
    resized = spectrogram[:,width * measure_number:width*(measure_number + 1)]
    if dim is None:
        return resized
    else:
        width, height = dim
        return zoom(resized, (height / resized.shape[0], width / resized.shape[1]))
    

if __name__ == '__main__':
    spectrogram, sample_rate = convertToFrequencyDomain('C:\\Users\\ikamo\\Downloads\\bass_house_sample.ogg', None)
    plt.figure(figsize=(4, 4))
    for i in range(9):
        librosa.display.specshow(getMeasure(spectrogram, 9, i, (32,32)), sr=sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        #plt.show()
