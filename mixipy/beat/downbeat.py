import librosa
import numpy as np
from scipy.stats import entropy, mode
from scipy.signal import hanning, fftconvolve
from scipy.fftpack import fft

EPS = 1e-10  # A small epsilon value to avoid log(0)

def adaptive_threshold(data):
    sz = len(data)
    if sz == 0:
        return data

    smoothed = np.copy(data)
    p_pre = 8
    p_post = 7

    for i in range(sz):
        first = max(0, i - p_pre)
        last = min(sz - 1, i + p_post)
        smoothed[i] = np.mean(data[first:last + 1])

    for i in range(sz):
        data[i] -= smoothed[i]
        if data[i] < 0.0:
            data[i] = 0.0
    return data


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

def find_downbeats(audio, beats, increment, factor, beatframesize, bpb):
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
    if len(audio) == 0:
        return []

    oldspec = np.zeros(beatframesize // 2)
    beatsd = []

    for i in range(len(beats) - 1):
        beatstart = int((beats[i] * increment) // factor)
        beatend = int((beats[i + 1] * increment) // factor)
        if beatend >= len(audio):
            beatend = len(audio) - 1
        if beatend < beatstart:
            beatend = beatstart

        beatlen = beatend - beatstart
        beatframe = audio[beatstart:beatend] * hanning(beatlen)
        beatframe = np.pad(beatframe, (0, beatframesize - len(beatframe)), mode='constant')

        fft_out = fft(beatframe)
        newspec = np.abs(fft_out[:beatframesize // 2])

        newspec = adaptive_threshold(newspec)

        if i > 0:
            beatsd.append(measure_spec_diff(oldspec, newspec))

        oldspec = np.copy(newspec)

    timesig = bpb if bpb != 0 else 4
    dbcand = np.zeros(timesig)

    for beat in range(timesig):
        count = 0
        for example in range(beat - 1, len(beatsd), timesig):
            if example < 0:
                continue
            dbcand[beat] += (beatsd[example]) / timesig
            count += 1
        if count > 0:
            dbcand[beat] /= count

    dbind = np.argmax(dbcand)
    downbeats = [i for i in range(dbind, len(beats), timesig)]

    return downbeats

def measure_spec_diff(oldspec, newspec):
    # JENSEN-SHANNON DIVERGENCE BETWEEN SPECTRAL FRAMES
    SPECSIZE = 512  # ONLY LOOK AT FIRST 512 SAMPLES OF SPECTRUM.
    if SPECSIZE > len(oldspec) // 4:
        SPECSIZE = len(oldspec) // 4
    
    SD = 0.0

    # Add EPS to avoid taking log of 0
    newspec = newspec[:SPECSIZE] + EPS
    oldspec = oldspec[:SPECSIZE] + EPS
    
    sumnew = np.sum(newspec)
    sumold = np.sum(oldspec)
    
    # Normalize the spectra
    newspec /= sumnew
    oldspec /= sumold
    
    # Replace any remaining zeros with ones (after normalization, this shouldn't happen, but just in case)
    newspec = np.where(newspec == 0, 1.0, newspec)
    oldspec = np.where(oldspec == 0, 1.0, oldspec)
    
    # Jensen-Shannon calculation
    m = 0.5 * (oldspec + newspec)
    SD = np.sum(-m * np.log(m) + 0.5 * (oldspec * np.log(oldspec)) + 0.5 * (newspec * np.log(newspec)))
    
    return SD

def __beat_track_dp(localscore, period, tightness, time_signature, downbeat_frames):
    """Core dynamic program for beat tracking with time signature and downbeats"""
    backlink = np.zeros_like(localscore, dtype=int)
    cumscore = np.zeros_like(localscore)

    # Search range for previous beat
    window = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)

    # Make a score window, which begins biased toward start_bpm and skewed
    if tightness <= 0:
        raise ParameterError("tightness must be strictly positive")

    txwt = -tightness * (np.log(-window / period) ** 2)

    # Are we on the first beat?
    first_beat = True
    for i, score_i in enumerate(localscore):
        # Are we reaching back before time 0?
        z_pad = np.maximum(0, min(-window[0], len(window)))

        # Search over all possible predecessors
        candidates = txwt.copy()
        candidates[z_pad:] = candidates[z_pad:] + cumscore[window[z_pad:]]

        # Adjust scores based on time signature and downbeats
        if i in downbeat_frames:
            # Encourage the selection of downbeats
            candidates += 0.1 * localscore.max()
        else:
            # Penalize non-downbeat locations
            beat_location_in_measure = (i - downbeat_frames[downbeat_frames <= i][-1]) % time_signature
            if beat_location_in_measure != 0:
                candidates -= 0.05 * localscore.max()

        # Find the best preceding beat
        beat_location = np.argmax(candidates)

        # Add the local score
        cumscore[i] = score_i + candidates[beat_location]

        # Special case the first onset.  Stop if the localscore is small
        if first_beat and score_i < 0.01 * localscore.max():
            backlink[i] = -1
        else:
            backlink[i] = window[beat_location]
            first_beat = False

        # Update the time range
        window = window + 1

    return backlink, cumscore