import librosa
import math
import numpy as np

def beat_track(onset_envelope, sr, hop_length, input_tempo=120, constrain_tempo=False, alpha=0.9, tightness=4.0):
    # Calculate beat periods and tempi
    beat_periods, tempi = calculate_beat_period(onset_envelope, input_tempo, constrain_tempo)
    
    # Get the best tempo
    tempo = get_best_tempo(tempi, scores)
    
    # Calculate beat locations
    beats = calculate_beats(onset_envelope, beat_periods, alpha, tightness)
    
    # Convert beat locations to timestamps if needed
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    
    return tempo, beat_times


def calculate_beat_period(onset_envelope, sr,  input_tempo=120, constrain_tempo=False, win_length=512, hop_length=128):
    """Calculate the beat periods and tempi from the onset envelope using a comb filter bank and Viterbi decoding."""

    wv_len = hop_length
    rayparam = (60 * sr / win_length) / input_tempo

    # Make Rayleigh weighting curve or Gaussian weighting curve
    if constrain_tempo:
        wv = [math.exp((-1. * (i - rayparam) ** 2) / (2. * (rayparam / 4.) ** 2)) for i in range(wv_len)]
    else:
        wv = [(i / rayparam ** 2) * math.exp((-1. * (-i) ** 2) / (2. * rayparam ** 2)) for i in range(wv_len)]

    rcf_matrix = []
    onset_envelope_len = len(onset_envelope)

    # Main loop for beat period calculation
    for i in range(0, onset_envelope_len - win_length, hop_length):
        onset_frame = onset_envelope[i:i + win_length]
        rcf = get_rcf(onset_frame, wv)
        rcf_matrix.append(rcf)

    beat_periods, tempi = viterbi_decode(rcf_matrix, wv)

    return beat_periods, tempi

EPS = 0.0000008

def adaptive_threshold(data):
    sz = len(data)
    if sz == 0:
        return data  # Return the unmodified data if it's empty

    smoothed = np.zeros(sz)
    p_pre = 8
    p_post = 7

    for i in range(sz):
        first = max(0, i - p_pre)
        last = min(sz, i + p_post + 1)  # Adjust for Python's exclusive end in slicing
        smoothed[i] = np.mean(data[first:last])

    for i in range(sz):
        data[i] -= smoothed[i]
        if data[i] < 0.0:
            data[i] = 0.0
    
    return data  # Explicitly return the modified data

def get_rcf(onset_frame, wv):
    """Calculate the resonator comb filter (RCF) for a given detection function frame and weighting vector."""
    
    dfframe = adaptive_threshold(onset_frame.copy())  # Use a copy to avoid modifying the original data
    dfframe_len = len(dfframe)
    rcf = [0.0] * len(wv)

    # Calculate autocorrelation function (ACF)
    acf = [0.0] * dfframe_len
    for lag in range(dfframe_len):
        acf[lag] = sum(dfframe[n] * dfframe[n + lag] for n in range(dfframe_len - lag)) / (dfframe_len - lag)

    # Apply comb filtering
    numelem = 4
    for i in range(2, len(wv)):
        for a in range(1, numelem + 1):
            for b in range(1 - a, a):
                if (a * i + b) - 1 < len(acf):
                    rcf[i - 1] += (acf[(a * i + b) - 1] * wv[i - 1]) / (2. * a - 1.)

    # Apply adaptive threshold to rcf
    rcf = adaptive_threshold(rcf)

    # Normalize rcf to sum to unity
    rcfsum = sum(rcf) + EPS
    rcf = [x / rcfsum for x in rcf]

    return rcf

def viterbi_decode(rcfmat, wv, sr, hop_length):
    """
    Perform Viterbi decoding on the RCF matrix to identify the best beat path.

    Parameters:
    - rcfmat: The matrix of resonator comb filter outputs.
    - wv: The weighting vector used in RCF calculation.
    - sr: Sample rate of the audio signal.
    - hop_length: The hop length used in onset envelope calculation, equivalent to df_increment.

    Returns:
    - beat_period: The best beat period path identified by Viterbi decoding.
    - tempi: The tempi corresponding to the beat periods.
    """
    # Ensure input validity
    if not rcfmat or not wv:
        raise ValueError("Input matrices cannot be empty.")

    wv_len = len(wv)
    tmat = np.zeros((wv_len, wv_len))
    sigma = 8.0

    # Fill transition matrix with Gaussian values
    for i in range(20, wv_len - 20):
        mu = float(i)
        tmat[i, 20:wv_len - 20] = np.exp(-1.0 * np.square(np.arange(20, wv_len - 20) - mu) / (2 * sigma ** 2))

    T, Q = len(rcfmat), len(rcfmat[0])
    if T < 2:
        return np.array([]), np.array([])

    delta = np.zeros((T, Q))
    psi = np.zeros((T, Q), dtype=int)

    # Initialize first column of delta
    delta[0] = wv * rcfmat[0]
    delta[0] /= (np.sum(delta[0]) + EPS)

    # Viterbi forward pass
    for t in range(1, T):
        for j in range(Q):
            tmp_vec = delta[t - 1] * tmat[j]
            psi[t, j] = np.argmax(tmp_vec)
            delta[t, j] = tmp_vec[psi[t, j]] * rcfmat[t][j]
        delta[t] /= (np.sum(delta[t]) + EPS)

    # Traceback
    bestpath = np.zeros(T, dtype=int)
    bestpath[-1] = np.argmax(delta[-1])

    for t in range(T - 2, -1, -1):
        bestpath[t] = psi[t + 1, bestpath[t + 1]]

    # Construct beat_period and tempi from bestpath
    beat_period = np.zeros_like(rcfmat[0])
    for i, bp in enumerate(bestpath):
        beat_period[i * 128: (i + 1) * 128] = bp

    beat_period[-128:] = bestpath[-1]  # Handle potential size mismatch
    tempi = (60.0 * sr / hop_length) / beat_period

    return beat_period, tempi


def calculate_beats(df, beat_period, alpha=0.9, tightness=4.0):
    """
    Calculate beat locations from the detection function and beat periods.
    
    Parameters:
    - df: The detection function.
    - beat_period: Beat periods calculated from the Viterbi decoding.
    - alpha: The weighting between tempo consistency and the detection function.
    - tightness: Controls the tightness of the tempo consistency.

    Returns:
    - beats: The final calculated beat positions.
    """
    if not df or not beat_period:
        return []

    df_len = len(df)
    cumscore = np.zeros(df_len)
    backlink = np.zeros(df_len, dtype=int)
    localscore = np.array(df)

    # Main loop
    for i in range(df_len):
        prange_min = int(max(-2 * beat_period[i], -df_len))
        prange_max = int(min(round(-0.5 * beat_period[i]), df_len - 1))
        txwt_len = prange_max - prange_min + 1
        txwt = np.zeros(txwt_len)
        scorecands = np.zeros(txwt_len)

        for j in range(txwt_len):
            txwt[j] = np.exp(-0.5 * ((tightness * np.log((round(2 * beat_period[i]) - j) / beat_period[i])) ** 2))
            cscore_ind = i + prange_min + j
            if 0 <= cscore_ind < df_len:
                scorecands[j] = txwt[j] * cumscore[cscore_ind]

        vv = np.max(scorecands)
        xx = np.argmax(scorecands)

        cumscore[i] = alpha * vv + (1 - alpha) * localscore[i]
        backlink[i] = i + prange_min + xx

    # Traceback to find the best path
    bestpath = [np.argmax(cumscore)]
    while backlink[bestpath[-1]] != -1 and backlink[bestpath[-1]] != bestpath[-1]:
        bestpath.append(backlink[bestpath[-1]])

    bestpath.reverse()
    beats = np.array(bestpath, dtype=int)

    return beats