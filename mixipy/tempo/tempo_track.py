import math
import numpy as np

EPS = 0.0000008  # just some arbitrary small number

class TempoTrack:
    def __init__(self, sample_rate, df_increment):
        self.sample_rate = sample_rate
        self.increment = df_increment


    def calculate_beat_period(self, df, input_tempo=120.0, constrain_tempo=False):
        """Calculate the beat periods and tempi from the detection function, with optional input tempo and tempo constraint."""

        beat_period, tempi = self._calculate_beat_period(df, input_tempo, constrain_tempo)
        return beat_period, tempi


    def calculate_beats(self, df, beat_period, alpha=0.9, tightness=4.0):
        """Calculate the beat locations from the detection function and beat periods, with adjustable alpha and tightness parameters."""

        beats = self._calculate_beats(df, beat_period, alpha, tightness)
        return beats


    def filter_df(self, df):
        """Apply a forward-backward butterworth filter to the detection function."""

        df_len = len(df)
        a = [1.0000, -0.3695, 0.1958]
        b = [0.2066, 0.4131, 0.2066]
        lp_df = [0.0] * df_len

        inp1, inp2, out1, out2 = 0.0, 0.0, 0.0, 0.0

        # Forward filtering
        for i in range(df_len):
            lp_df[i] = b[0] * df[i] + b[1] * inp1 + b[2] * inp2 - a[1] * out1 - a[2] * out2
            inp2, inp1 = inp1, df[i]
            out2, out1 = out1, lp_df[i]

        # Copy forward filtering to df, but time-reversed
        df = lp_df[::-1]

        lp_df = [0.0] * df_len
        inp1, inp2, out1, out2 = 0.0, 0.0, 0.0, 0.0

        # Backward filtering on time-reversed df
        for i in range(df_len):
            lp_df[i] = b[0] * df[i] + b[1] * inp1 + b[2] * inp2 - a[1] * out1 - a[2] * out2
            inp2, inp1 = inp1, df[i]
            out2, out1 = out1, lp_df[i]

        # Write the re-reversed (i.e., forward) version back to df
        df = lp_df[::-1]

        return df


    def _calculate_beat_period(self, df, input_tempo, constrain_tempo, winlen=512, step = 128):
        """Calculate the beat periods and tempi from the detection function using a comb filter bank and Viterbi decoding."""

        wv_len = 128
        rayparam = (60 * 44100 / 512) / input_tempo

        # Make Rayleigh weighting curve or Gaussian weighting curve
        if constrain_tempo:
            wv = [math.exp((-1. * (i - rayparam) ** 2) / (2. * (rayparam / 4.) ** 2)) for i in range(wv_len)]
        else:
            wv = [(i / rayparam ** 2) * math.exp((-1. * (-i) ** 2) / (2. * rayparam ** 2)) for i in range(wv_len)]

        rcfmat = []
        df_len = len(df)

        # Main loop for beat period calculation
        for i in range(0, df_len - winlen, step):
            dfframe = df[i:i + winlen]
            rcf = self._get_rcf(dfframe, wv)
            rcfmat.append(rcf)

        beat_period, tempi = self._viterbi_decode(rcfmat, wv)

        return beat_period, tempi

    def _get_rcf(self, dfframe, wv):
        """Calculate the resonator comb filter (RCF) for a given detection function frame and weighting vector."""

        dfframe = self._adaptive_threshold(dfframe)
        dfframe_len = len(dfframe)
        rcf_len = len(wv)

        acf = [0.0] * dfframe_len
        for lag in range(dfframe_len):
            acf[lag] = sum(dfframe[n] * dfframe[n + lag] for n in range(dfframe_len - lag)) / (dfframe_len - lag)

        # Apply comb filtering
        numelem = 4
        rcf = [0.0] * rcf_len
        for i in range(2, rcf_len):
            for a in range(1, numelem + 1):
                for b in range(1 - a, a):
                    rcf[i - 1] += (acf[(a * i + b) - 1] * wv[i - 1]) / (2. * a - 1.)

        # Apply adaptive threshold to rcf
        rcf = self._adaptive_threshold(rcf)

        rcfsum = sum(rcf) + EPS
        rcf = [x / rcfsum for x in rcf]

        return rcf

    def _viterbi_decode(self, rcfmat, wv):
        wv_len = len(wv)

        # Make transition matrix
        tmat = [[0.0] * wv_len for _ in range(wv_len)]
        sigma = 8.0
        for i in range(20, wv_len - 20):
            for j in range(20, wv_len - 20):
                mu = float(i)
                tmat[i][j] = math.exp((-1. * (j - mu) ** 2) / (2. * sigma ** 2))

        delta = [[0.0] * len(rcfmat[0]) for _ in range(len(rcfmat))]
        psi = [[0] * len(rcfmat[0]) for _ in range(len(rcfmat))]

        T = len(delta)
        if T < 2:
            return [], []

        Q = len(delta[0])

        # Initialize first column of delta
        for j in range(Q):
            delta[0][j] = wv[j] * rcfmat[0][j]
            psi[0][j] = 0

        deltasum = sum(delta[0])
        delta[0] = [x / (deltasum + EPS) for x in delta[0]]

        for t in range(1, T):
            for j in range(Q):
                tmp_vec = [delta[t - 1][i] * tmat[j][i] for i in range(Q)]
                delta[t][j] = max(tmp_vec)
                psi[t][j] = tmp_vec.index(delta[t][j])
                delta[t][j] *= rcfmat[t][j]

            # Normalize current delta column
            deltasum = sum(delta[t])
            delta[t] = [x / (deltasum + EPS) for x in delta[t]]

        bestpath = [0] * T
        bestpath[T - 1] = delta[T - 1].index(max(delta[T - 1]))

        # Backtrace through index of maximum values in psi
        for t in range(T - 2, 0, -1):
            bestpath[t] = psi[t + 1][bestpath[t + 1]]

        bestpath[0] = psi[1][bestpath[1]]

        beat_period = [0.0] * len(rcfmat[0])
        tempi = []
        lastind = 0
        for i in range(T):
            step = 128
            beat_period[i * step:(i + 1) * step] = [bestpath[i]] * step
            lastind = (i + 1) * step

        # Fill in the last values
        beat_period[lastind:] = [beat_period[lastind - 1]] * (len(beat_period) - lastind)

        for i in range(len(beat_period)):
            tempi.append((60. * self.sample_rate / self.increment) / beat_period[i])

        return beat_period, tempi

    def _calculate_beats(self, df, beat_period, alpha, tightness):
        """Calculate the beat locations from the detection function and beat periods using dynamic programming."""
        if not df or not beat_period:
            return []

        df_len = len(df)

        cumscore = [0.0] * df_len
        backlink = [-1] * df_len
        localscore = df.copy()

        for i in range(df_len):
            prange_min = -2 * int(beat_period[i])
            prange_max = round(-0.5 * beat_period[i])

            txwt_len = prange_max - prange_min + 1
            txwt = [0.0] * txwt_len
            scorecands = [0.0] * txwt_len

            for j in range(txwt_len):
                mu = beat_period[i]
                txwt[j] = math.exp(-0.5 * (tightness * math.log((round(2 * mu) - j) / mu)) ** 2)

                cscore_ind = i + prange_min + j
                if cscore_ind >= 0:
                    scorecands[j] = txwt[j] * cumscore[cscore_ind]

            vv = max(scorecands)
            xx = scorecands.index(vv)

            cumscore[i] = alpha * vv + (1. - alpha) * localscore[i]
            backlink[i] = i + prange_min + xx

        startpoint = df_len - int(beat_period[-1]) + np.argmax(cumscore[-int(beat_period[-1]):])
        if startpoint >= len(backlink):
            startpoint = len(backlink) - 1

        ibeats = [startpoint]
        while backlink[ibeats[-1]] > 0:
            b = ibeats[-1]
            if backlink[b] == b:
                break
            ibeats.append(backlink[b])

        beats = ibeats[::-1]
        return beats

    @staticmethod
    def adapt_thresh(df):
        """Apply adaptive thresholding to the input array (placeholder method)."""

        # Replace with the actual implementation if available
        return arr