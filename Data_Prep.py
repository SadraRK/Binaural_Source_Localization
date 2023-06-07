import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import librosa
from scipy.io import wavfile
import soundfile as sf
from sklearn.decomposition import NMF
from scipy.io import savemat

def center(X):
    X = np.array(X)
    mean = X.mean(axis=1, keepdims=True)
    return X - mean

def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten

def seperate_sources(n_components, signal, og_signal):  # Apply NMF algorithm to separate sounds
    stft_2d = signal.reshape(-1, signal.shape[-1])
    model = NMF(n_components=n_components, init='nndsvda', max_iter=300)
    W = model.fit_transform(np.abs(stft_2d.T))
    H = model.components_.reshape(n_components, 2, -1)

    # Reconstruct separated signals from NMF components
    source_signals = []
    source_signals_stft = []
    for i in range(n_components):
        source_stft = np.zeros((signal.shape[0], signal.shape[1], signal.shape[2]))
        source_stft[0, :, :] = np.transpose(np.outer(W[:, i], H[i, 0, :])[:, :, np.newaxis])
        source_stft[1, :, :] = np.transpose(np.outer(W[:, i], H[i, 1, :])[:, :, np.newaxis])
        source_signal = librosa.istft(source_stft)
        source_signal[0, :] = source_signal[0, :] / np.max(np.abs(source_signal[0, :]))
        source_signal[1, :] = source_signal[1, :] / np.max(np.abs(source_signal[1, :]))
        source_signals.append(source_signal)
        source_signals_stft.append(source_stft)
    min_length = min(len(source_signal) for source_signal in source_signals)
    source_signals = [source_signal[:min_length] for source_signal in source_signals]

    audio_windowed = np.zeros((np.shape(source_signals)[0], 2, len(og_signal[0])))
    for i in range(np.shape(source_signals)[0]):
        idx_temp_left = np.where(source_signals[i][0] >= 0.6)
        idx_temp_right = np.where(source_signals[i][1] >= 0.6)
        window = 25000
        mask_left = np.zeros((len(og_signal[0])))
        mask_right = np.zeros((len(og_signal[1])))

        if (idx_temp_left[0][0] - window < 0) and (idx_temp_left[0][-1] + window > len(og_signal[0])):
            mask_left = 1
        elif (idx_temp_left[0][0] - window < 0) and (idx_temp_left[0][-1] + window <= len(og_signal[0])):
            mask_left[0:(idx_temp_left[0][-1] + window)] = 1
        elif (idx_temp_left[0][0] - window >= 0) and (idx_temp_left[0][-1] + window > len(og_signal[0])):
            mask_left[(idx_temp_left[0][0] - window):] = 1
        else:
            mask_left[(idx_temp_left[0][0] - window):(idx_temp_left[0][-1] + window)] = 1

        if (idx_temp_right[0][0] - window < 0) and (idx_temp_right[0][-1] + window > len(og_signal[1])):
            mask_right = 1
        elif (idx_temp_right[0][0] - window < 0) and (idx_temp_right[0][-1] + window <= len(og_signal[1])):
            mask_right[0:(idx_temp_right[0][-1] + window)] = 1
        elif (idx_temp_right[0][0] - window >= 0) and (idx_temp_right[0][-1] + window > len(og_signal[1])):
            mask_right[(idx_temp_right[0][0] - window):] = 1
        else:
            mask_right[(idx_temp_right[0][0] - window):(idx_temp_right[0][-1] + window)] = 1

        audio_windowed_left = og_signal[0] * mask_left
        audio_windowed_left = audio_windowed_left / np.max(np.abs(audio_windowed_left))
        audio_windowed_right = og_signal[1] * mask_right
        audio_windowed_right = audio_windowed_right / np.max(np.abs(audio_windowed_right))
        audio_windowed[i, :, :] = np.vstack([audio_windowed_left, audio_windowed_right])
    fig = plt.figure()
    plt.plot(np.linspace(0, len(source_signals[0][0, :]), len(source_signals[0][0, :])), source_signals[0][0, :])
    plt.plot(np.linspace(0, len(source_signals[1][0, :]), len(source_signals[1][0, :])), source_signals[1][0, :])
    plt.plot(np.linspace(0, len(source_signals[2][0, :]), len(source_signals[2][0, :])), source_signals[2][0, :])
    plt.plot(np.linspace(0, len(source_signals[3][0, :]), len(source_signals[3][0, :])), source_signals[3][0, :])
    plt.plot(np.linspace(0, len(source_signals[4][0, :]), len(source_signals[4][0, :])), source_signals[4][0, :])
    plt.title("Seperated Sounds of Channel 1")
    plt.legend(['S0', 'S1', 'S2', 'S3', 'S4'])
    plt.ylabel('Amplitude (a.u.)')
    plt.xlabel('Time (s)')
    fig.tight_layout()
    plt.show()

    fig = plt.figure()
    plt.plot(np.linspace(0, len(source_signals[0][1, :]), len(source_signals[0][1, :])), source_signals[0][1, :])
    plt.plot(np.linspace(0, len(source_signals[1][1, :]), len(source_signals[1][1, :])), source_signals[1][1, :])
    plt.plot(np.linspace(0, len(source_signals[2][1, :]), len(source_signals[2][1, :])), source_signals[2][1, :])
    plt.plot(np.linspace(0, len(source_signals[3][1, :]), len(source_signals[3][1, :])), source_signals[3][1, :])
    plt.plot(np.linspace(0, len(source_signals[4][1, :]), len(source_signals[4][1, :])), source_signals[4][1, :])
    plt.title("Seperated Sounds of Channel 2")
    plt.legend(['S0', 'S1', 'S2', 'S3', 'S4'])
    plt.ylabel('Amplitude (a.u.)')
    plt.xlabel('Time (s)')
    fig.tight_layout()
    plt.show()
    return audio_windowed

def compute_ITD(signal):
    itd = np.zeros((np.shape(signal)[0]))
    angle = np.zeros((np.shape(signal)[0]))
    for i in range(np.shape(signal)[0]):
        # cross_correlation = np.fft.irfft(signal[i][0] * np.conj(signal[(i + 1) % 4][1]))
        cross_correlation = np.correlate(signal[i][0], signal[(i + 1) % 4][1], mode='same')
        min_length = len(signal[i][0])
        cross_correlation = cross_correlation[:min_length]
        itd[i] = np.argmax(cross_correlation) - (np.shape(signal)[2] - 1) // 2  # in samples
        angle[i] = np.arctan2(itd[i] / sr * c, d) * 180 / np.pi  # in degrees
        print(f"ITD: {itd[i]:.6f} s, Azimuth angle: {angle[i]:.2f} degrees")
        fig = plt.figure()
        plt.plot(cross_correlation)
        plt.title("Cross-correlation between left and right ear")
        plt.show
    return itd, angle

def compute_ILD(signal):
    ild = np.zeros((np.shape(signal)[0]))
    angle = np.zeros((np.shape(signal)[0]))
    for i in range(np.shape(signal)[0]):
        idx_left = np.where(np.abs(signal[i][0]) >= 0.45)
        idx_right = np.where(np.abs(signal[i][1]) >= 0.45)
        # print(idx_left, idx_right)
        alpha = 1 - (np.mean(np.abs(signal[i][0, idx_left])) / np.mean(np.abs(signal[i][1, idx_right])))**2
        ild[i] = (1.01 * alpha)/(0.4 - 0.2*alpha)
        if ild[i] > 0:
            azimuth = np.arccos(ild[i])
        elif ild[i] < 0:
            azimuth = -np.arccos(ild[i])
        else:
            azimuth = 0
        azimuth_degrees = np.rad2deg(azimuth)
        angle[i] = azimuth_degrees
        print(f"ILD: {ild[i]:.6f} s, Azimuth angle: {angle[i]:.2f} degrees")
    return ild, angle

sns.set(rc={'figure.figsize':(11.7,8.27)})
## Parameters for sound source localization
c = 343 # speed of sound in m/s
d = 0.2 # distance between microphones in meters
theta = []

## Load Binaural audio files
db_path = 'Binaural_Dataset'
audio, sr = librosa.load(db_path + '/Audio/split1_ov1_1.wav', sr=None, mono=False)
audio_processed = whitening(center(audio))
time_series = np.linspace(0, len(audio[0]), len(audio[0])) / sr

## Perform Spectral Analysis on audio files -- fft
num_points = int(1.44e6)
stft = librosa.stft(audio_processed)
fft_freq = np.fft.rfftfreq(n=num_points, d=1/sr)*1e-3
fft_left = np.fft.fft(audio[0], n=num_points)
fft_left_mag = np.abs(fft_left[0:int(num_points/2)])
fft_right = np.fft.fft(audio[1], n=num_points)
fft_right_mag = np.abs(fft_right[0:int(num_points/2)])

## Perform Blind-Source_Separation on audio files -- NMF
audio_windowed = seperate_sources(n_components=5, signal=stft, og_signal=audio)
# compute_ILD(source_signals)

# Save audio files
# for i in range(5):
#     savemat(db_path+'/Processed/sub_1_'+str(i)+'.mat', mdict={'audio': audio_windowed[i]})

## Plot Data
fig = plt.figure()
plt.plot(time_series, audio[0], time_series, audio[1])
plt.title("Sound Wave")
plt.legend(['Channel 1/Left', 'Channel 2/Right'])
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
fig.tight_layout()
plt.show()

fig = plt.figure()
plt.plot(fft_freq[0:-1], fft_left_mag, fft_freq[0:-1], fft_right_mag)
plt.title("Sound Wave")
plt.legend(['Channel 1/Left', 'Channel 2/Right'])
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (KHz)')
fig.tight_layout()
plt.show()

fig = plt.figure()
plt.plot(time_series, audio_windowed[0][0], time_series, audio_windowed[0][1])
plt.title("Seperated Sound 1")
plt.legend(['Channel 1/Left', 'Channel 2/Right'])
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (s)')
fig.tight_layout()
plt.show()

fig = plt.figure()
plt.plot(time_series, audio_windowed[1][0], time_series, audio_windowed[1][1])
plt.title("Seperated Sound 2")
plt.legend(['Channel 1/Left', 'Channel 2/Right'])
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (s)')
fig.tight_layout()
plt.show()

fig = plt.figure()
plt.plot(time_series, audio_windowed[2][0], time_series, audio_windowed[2][1])
plt.title("Seperated Sound 3")
plt.legend(['Channel 1/Left', 'Channel 2/Right'])
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (s)')
fig.tight_layout()
plt.show()

fig = plt.figure()
plt.plot(time_series, audio_windowed[3][0], time_series, audio_windowed[3][1])
plt.title("Seperated Sound 4")
plt.legend(['Channel 1/Left', 'Channel 2/Right'])
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (s)')
fig.tight_layout()
plt.show()

fig = plt.figure()
plt.plot(time_series, audio_windowed[4][0], time_series, audio_windowed[4][1])
plt.title("Seperated Sound 5")
plt.legend(['Channel 1/Left', 'Channel 2/Right'])
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Time (s)')
fig.tight_layout()
plt.show()
