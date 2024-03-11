import librosa
import numpy as np
from scipy.stats import entropy

def estimate_time_signature(beat_times, sr, hop_length):
    """
    Estimates the time signature based on the beat times.

    Args:
        beat_times (np.ndarray): The beat times in seconds.
        sr (int): The sampling rate of the audio signal.
        hop_length (int): The hop length used to compute the beat times.

    Returns:
        int: The estimated time signature.
    """
    # Compute the inter-beat intervals (IBIs) in frames
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
    ibis_frames = np.diff(beat_frames)

    # Quantize the IBIs to the nearest beat period
    tempo, _ = librosa.beat.beat_tracker(y=None, sr=sr, onset_envelope=None, hop_length=hop_length)
    beat_period_frames = int(round(60 * sr / tempo / hop_length))
    quantized_ibis = np.round(ibis_frames / beat_period_frames) * beat_period_frames

    # Calculate the mode of the quantized IBIs
    time_signature = int(mode(quantized_ibis / beat_period_frames)[0][0])

    return time_signature

def find_downbeats(y, sr, hop_length, beat_times, time_signature):
    """
    Finds the downbeat locations using the spectral difference approach.

    Args:
        y (np.ndarray): The audio signal.
        sr (int): The sampling rate of the audio signal.
        hop_length (int): The hop length used for analysis.
        beat_times (np.ndarray): The beat times in seconds.
        time_signature (int): The time signature of the audio signal.

    Returns:
        np.ndarray: The downbeat times in seconds.
    """
    # Beat frame size is the next power of two up from 1.3 seconds at the downsampled rate
    beat_frame_size = int(librosa.time_to_frames(1.3, sr=sr, hop_length=hop_length))
    beat_frame_size = 2 ** np.ceil(np.log2(beat_frame_size))

    # Convert beat times to frames
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)

    # Compute the spectral difference between beat frames
    beat_sd = []
    old_spec = None
    for i in range(len(beat_frames) - 1):
        start = beat_frames[i]
        end = beat_frames[i + 1]
        beat_frame = y[start:end]

        # Apply Hanning window
        beat_len = end - start
        window = np.hanning(beat_len)
        beat_frame *= window

        # Compute FFT
        fft_frame = np.fft.rfft(beat_frame, n=beat_frame_size)
        new_spec = np.abs(fft_frame)[:beat_frame_size // 2]

        # Adaptive thresholding
        new_spec[new_spec < np.max(new_spec) / 10] = 0

        # Calculate Jensen-Shannon divergence between new and old spectral frames
        if old_spec is not None:
            beat_sd.append(jensen_shannon_divergence(old_spec, new_spec))

        old_spec = new_spec

    # Find the beat transition with the greatest spectral change
    dbcand = np.zeros(time_signature)
    for beat in range(time_signature):
        count = 0
        for example in range(beat - 1, len(beat_sd), time_signature):
            if example < 0:
                continue
            dbcand[beat] += beat_sd[example] / time_signature
            count += 1
        if count > 0:
            dbcand[beat] /= count

    # First downbeat is the beat at the index of the maximum value of dbcand
    dbind = np.argmax(dbcand)

    # Remaining downbeats are at time_signature intervals from the first
    downbeat_times = [beat_times[dbind]]
    for i in range(dbind + time_signature, len(beat_times), time_signature):
        downbeat_times.append(beat_times[i])

    return np.array(downbeat_times)

def jensen_shannon_divergence(p, q):
    """
    Calculates the Jensen-Shannon divergence between two probability distributions.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The Jensen-Shannon divergence between `p` and `q`.
    """
    m = 0.5 * (p + q)
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))
    return jsd