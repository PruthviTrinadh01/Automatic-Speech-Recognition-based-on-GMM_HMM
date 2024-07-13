#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import scipy.fftpack
import librosa
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt


# In[10]:


def pre_emphasis(signal, pre_emphasis_coeff=0.97):
    return np.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])


# In[11]:


# Step 1: Frames
def framing(signal, frame_size, frame_stride, sample_rate):
    frame_length = int(frame_size * sample_rate)
    frame_step = int(frame_stride * sample_rate)
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames


# In[12]:


# Step 2: Windowing
def hamming_window(frame_length):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))

def apply_window(frames):
    window = hamming_window(frames.shape[1])
    return frames * window


# In[13]:


# Step 3: Fourier Transform
def fft_magnitude(frames):
    frame_length = frames.shape[1]
    NFFT = 2 ** int(np.ceil(np.log2(frame_length)))  # Ensure FFT length is a power of two
    fft_results = np.fft.rfft(frames, NFFT)  # Compute the FFT
    magnitude_frames = np.abs(fft_results)  # Calculate the magnitude of the FFT results
    return magnitude_frames, NFFT


# In[14]:


# Step 4: Mel Filter Bank Processing
def hz_to_mel(hz):
    return 1127 * np.log1p(hz / 700)

# Hz from Mel scale frequency
def mel_to_hz(mel):
    return 700 * (np.expm1(mel / 1127))

# Mel filter bank
def create_mel_filterbank(sample_rate, NFFT, num_mel_filters=26):
    low_freq_mel = hz_to_mel(0)  # Lower bound of human hearing in Mel scale
    high_freq_mel = hz_to_mel(sample_rate / 2)  
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_mel_filters + 2)  
    hz_points = mel_to_hz(mel_points)  # Convert Mel points back to Hz
    bin_points = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)  
    
    filters = np.zeros((num_mel_filters, NFFT // 2 + 1))
    for i in range(1, num_mel_filters + 1):
        start, center, end = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        for j in range(start, center):
            filters[i - 1, j] = (j - start) / (center - start)
        for j in range(center, end):
            filters[i - 1, j] = (end - j) / (end - center)

    return filters

# Apply the Mel filter bank to the FFT magnitude frames
def apply_mel_filters(fft_magnitude_frames, sample_rate, NFFT, num_mel_filters=26):
    mel_filters = create_mel_filterbank(sample_rate, NFFT, num_mel_filters)
    mel_warped_spectra = np.dot(fft_magnitude_frames, mel_filters.T)
    return mel_warped_spectra

# Logarithmic scaling
def log_scale(mel_spectra):
    log_mel_spectra = np.log(mel_spectra + 1e-10)
    return log_mel_spectra


# In[15]:


# Step 5: DCT with the first 13 coefficients
def dct_and_keep_coefficients(log_mel_spectra, num_coefficients=13):
    # Compute the DCT (type II) of the log Mel spectra
    mfcc = scipy.fftpack.dct(log_mel_spectra, type=2, axis=1, norm='ortho')
    # Keep only the first 13 coefficients
    return mfcc[:, :num_coefficients]


# In[16]:


# combine all the methods to compute MFCC coefficients
def compute_mfcc(signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01, num_mel_filters=26, num_coefficients=13):
    emphasized_signal = pre_emphasis(signal)
    frames = framing(emphasized_signal, frame_size, frame_stride, sample_rate)
    windowed_frames = apply_window(frames)
    fft_magnitude_frames, NFFT = fft_magnitude(windowed_frames)
    mel_warped_spectra = apply_mel_filters(fft_magnitude_frames, sample_rate, NFFT, num_mel_filters)
    log_mel_spectra = np.log(mel_warped_spectra)
    mfcc = dct_and_keep_coefficients(log_mel_spectra, num_coefficients)
    return mfcc


# In[17]:


# Delta features
def compute_delta(mfcc_features, M=2):
    delta_features = np.zeros_like(mfcc_features)
    denominator = 2 * sum([tau ** 2 for tau in range(1, M + 1)])
    
    for t in range(M, len(mfcc_features) - M):
        for j in range(mfcc_features.shape[1]):
            numerator = sum([tau * mfcc_features[t + tau, j] for tau in range(-M, M+1)])
            delta_features[t, j] = numerator / denominator

    # Trim the delta features by removing the first and last M frames to match the original matrix size
    return delta_features[M:-M]

def concatenate_mfcc_deltas(mfcc_features, delta_features, M = 2):
    trimmed_mfcc_features = mfcc_features[M:-M]
    concatenated_features = np.concatenate((trimmed_mfcc_features, delta_features), axis=1)
    return concatenated_features


# In[19]:


# Recording and saving audio
def record_audio(duration, sample_rate=16000):
    """Record audio from the microphone."""
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording stopped.")
    return audio.flatten()

def save_wav(audio, sample_rate, name):
    """Save recorded audio."""
    audio_1 = audio.astype('float32')
    sf.write(name + '.wav', audio_1, sample_rate)


# In[62]:


# Plotting signal and MFCC coefficients
def plot_signal(signal, fs, duration):
    time = np.arange(0, duration, 1/fs)
    epsilon = 1e-10
    signal_db = 20 * np.log10(np.maximum(np.abs(signal), epsilon))
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal_db)
    plt.title('Signal Magnitude in dB vs. Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.show()
    
def plot_MFCC(concatenate_features):
    plt.figure(figsize=(10, 4))
    plt.imshow(concatenate_features.T, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar()  # Show the color scale
    plt.xlabel('Time Frame')
    plt.ylabel('MFCC + Delta')
    plt.title('Feature Matrix Visualization')
    plt.show()


# # Odessa

# In[56]:


fs = 16000
duration = 2
odessa = record_audio(duration = duration, sample_rate = fs)
save_wav(odessa, fs, 'Odessa')


# In[117]:


M = 2
mfcc_features = compute_mfcc(odessa, sample_rate=fs)
delta_mfcc = compute_delta(mfcc_features, M)
concatenate_features = concatenate_mfcc_deltas(mfcc_features, delta_mfcc, M)
print("MFCC shape:", mfcc_features.T.shape)
print("Delta shape:", delta_mfcc.T.shape)
print("MFCC shape:", concatenate_features.T.shape)


# In[116]:


plot_signal(odessa, fs, duration)
plot_MFCC(concatenate_features)


# # Turn on the lights

# In[66]:


fs = 16000
duration = 4
lights_on = record_audio(duration = duration, sample_rate = fs)
save_wav(lights_on, fs, 'lights_on')


# In[118]:


M = 2
mfcc_features = compute_mfcc(lights_on, sample_rate=fs)
delta_mfcc = compute_delta(mfcc_features, M)
concatenate_features = concatenate_mfcc_deltas(mfcc_features, delta_mfcc, M)
print("MFCC shape:", mfcc_features.T.shape)
print("Delta shape:", delta_mfcc.T.shape)
print("MFCC shape:", concatenate_features.T.shape)


# In[69]:


plot_signal(lights_on, fs, duration)
plot_MFCC(concatenate_features)


# # Turn off the lights

# In[70]:


fs = 16000
duration = 4
lights_off = record_audio(duration = duration, sample_rate = fs)
save_wav(lights_off, fs, 'lights_off')


# In[119]:


M = 2
mfcc_features = compute_mfcc(lights_off, sample_rate=fs)
delta_mfcc = compute_delta(mfcc_features, M)
concatenate_features = concatenate_mfcc_deltas(mfcc_features, delta_mfcc, M)
print("MFCC shape:", mfcc_features.T.shape)
print("Delta shape:", delta_mfcc.T.shape)
print("MFCC shape:", concatenate_features.T.shape)


# In[74]:


plot_signal(lights_off, fs, duration)
plot_MFCC(concatenate_features)


# # What time is it

# In[80]:


fs = 16000
duration = 4
time = record_audio(duration = duration, sample_rate = fs)
save_wav(time, fs, 'time')


# In[120]:


M = 2
mfcc_features = compute_mfcc(time, sample_rate=fs)
delta_mfcc = compute_delta(mfcc_features, M)
concatenate_features = concatenate_mfcc_deltas(mfcc_features, delta_mfcc, M)
print("MFCC shape:", mfcc_features.T.shape)
print("Delta shape:", delta_mfcc.T.shape)
print("MFCC shape:", concatenate_features.T.shape)


# In[83]:


plot_signal(time, fs, duration)
plot_MFCC(concatenate_features)


# # Play Music

# In[100]:


fs = 16000
duration = 2
play = record_audio(duration = duration, sample_rate = fs)
save_wav(play, fs, 'play')


# In[121]:


M = 2
mfcc_features = compute_mfcc(play, sample_rate=fs)
delta_mfcc = compute_delta(mfcc_features, M)
concatenate_features = concatenate_mfcc_deltas(mfcc_features, delta_mfcc, M)
print("MFCC shape:", mfcc_features.T.shape)
print("Delta shape:", delta_mfcc.T.shape)
print("MFCC shape:", concatenate_features.T.shape)


# In[103]:


plot_signal(play, fs, duration)
plot_MFCC(concatenate_features)


# # Stop Music

# In[112]:


fs = 16000
duration = 2
stop = record_audio(duration = duration, sample_rate = fs)
save_wav(stop, fs, 'stop')


# In[122]:


M = 2
mfcc_features = compute_mfcc(stop, sample_rate=fs)
delta_mfcc = compute_delta(mfcc_features, M)
concatenate_features = concatenate_mfcc_deltas(mfcc_features, delta_mfcc, M)
print("MFCC shape:", mfcc_features.T.shape)
print("Delta shape:", delta_mfcc.T.shape)
print("MFCC shape:", concatenate_features.T.shape)


# In[115]:


plot_signal(stop, fs, duration)
plot_MFCC(concatenate_features)

