"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""

import pyaudio
import wave
import numpy as np
from scipy.fftpack import dct, fft
from scipy.io import wavfile
import math

def melFilterBank(blockSize=4096, numCoefficients=64, numBands=40, minHz=200, maxHz=8000):
    numBands = int(numCoefficients)
    maxMel = int(freqToMel(maxHz))
    minMel = int(freqToMel(minHz))

    # Create a matrix for triangular filters, one row per filter
    filterMatrix = np.zeros((numBands, blockSize))

    melRange = np.array(xrange(numBands + 2))

    melCenterFilters = melRange * (maxMel - minMel) / (numBands + 1) + minMel

    # each array index represent the center of each triangular filter
    aux = np.log(1 + 1000.0 / 700.0) / 1000.0
    aux = (np.exp(melCenterFilters * aux) - 1) / 22050
    aux = 0.5 + 700 * blockSize * aux
    aux = np.floor(aux)  # Arredonda pra baixo
    centerIndex = np.array(aux, int)  # Get int values

    for i in xrange(numBands):
        start, centre, end = centerIndex[i:i + 3]
        k1 = np.float32(centre - start)
        k2 = np.float32(end - centre)
        up = (np.array(xrange(start, centre)) - start) / k1
        down = (end - np.array(xrange(centre, end))) / k2

        filterMatrix[i][start:centre] = up
        filterMatrix[i][centre:end] = down

    return filterMatrix.transpose()

def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)

def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)

def calc_fft(frame, NFFT = 127):
	frame = list(bytearray(frame)) # convert string representation of binary stream chunk into a byte array.
	#frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
	frame *= np.hamming(len(frame))
	mag_frame = np.absolute(np.fft.rfft(frame, NFFT))  # Magnitude of the FFT
	pow_frame = ((1.0 / NFFT) * ((mag_frame) ** 2))  # Power Spectrum
	filteredSpectrum = np.dot(pow_frame,melFilterBank().T) # MEL filter banks
	logSpectrum = np.log(filteredSpectrum)
	dctSpectrum = dct(logSpectrum, type=2)  # MFCC :)
	return dctSpectrum

CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
num_chunks = 0

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
    pow_fft = calc_fft(data)
    num_chunks += 1

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
print ('length of pow_fft: %i'%len(pow_fft))
print(pow_fft)
print('num_chunks: %i'%num_chunks)