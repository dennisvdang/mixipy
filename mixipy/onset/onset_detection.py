import numpy as np
import librosa

class DetectionFunction:
    """A class for calculating various onset detection functions on audio signals."""

    def __init__(self, config):
        self.m_dataLength = config['frameLength']
        self.m_halfLength = self.m_dataLength // 2 + 1
        self.m_DFType = config['DFType']
        self.m_stepSize = config['stepSize']
        self.m_dbRise = config['dbRise']
        self.m_whiten = config['adaptiveWhitening']
        self.m_whitenRelaxCoeff = config['whiteningRelaxCoeff']
        self.m_whitenFloor = config['whiteningFloor']
        
        if self.m_whitenRelaxCoeff < 0:
            self.m_whitenRelaxCoeff = 0.9997
        if self.m_whitenFloor < 0:
            self.m_whitenFloor = 0.01
        
        self.m_magHistory = np.zeros(self.m_halfLength)
        self.m_phaseHistory = np.zeros(self.m_halfLength)
        self.m_phaseHistoryOld = np.zeros(self.m_halfLength)
        self.m_magPeaks = np.zeros(self.m_halfLength)
        
        self.m_phaseVoc = PhaseVocoder(self.m_dataLength, self.m_stepSize)
        self.m_magnitude = np.zeros(self.m_halfLength)
        self.m_thetaAngle = np.zeros(self.m_halfLength)
        self.m_unwrapped = np.zeros(self.m_halfLength)
        
        self.m_window = np.hanning(self.m_dataLength)
        self.m_windowed = np.zeros(self.m_dataLength)
    
    def processTimeDomain(self, samples):
        self.m_windowed = samples * self.m_window
        self.m_magnitude, self.m_thetaAngle, self.m_unwrapped = self.m_phaseVoc.processTimeDomain(self.m_windowed)
        
        if self.m_whiten:
            self.whiten()
        
        return self.runDF()
    
    def processFrequencyDomain(self, reals, imags):
        """Process the given frequency domain data and return the DF value."""
        self.m_magnitude, self.m_thetaAngle, self.m_unwrapped = self.m_phaseVoc.processFrequencyDomain(reals, imags)
        
        if self.m_whiten:
            self.whiten()
        
        return self.runDF()
    
    def whiten(self):
        """Apply adaptive whitening to the magnitude spectrum."""
        for i in range(self.m_halfLength):
            m = self.m_magnitude[i]
            if m < self.m_magPeaks[i]:
                m = m + (self.m_magPeaks[i] - m) * self.m_whitenRelaxCoeff
            if m < self.m_whitenFloor:
                m = self.m_whitenFloor
            self.m_magPeaks[i] = m
            self.m_magnitude[i] /= m
    
    def runDF(self):
        """Run the selected DF algorithm and return its value."""
        if self.m_DFType == DF_HFC:
            return self.HFC(self.m_halfLength, self.m_magnitude)
        elif self.m_DFType == DF_SPECDIFF:
            return self.specDiff(self.m_halfLength, self.m_magnitude)
        elif self.m_DFType == DF_PHASEDEV:
            return self.phaseDev(self.m_halfLength, self.m_thetaAngle)
        elif self.m_DFType == DF_COMPLEXSD:
            return self.complexSD(self.m_halfLength, self.m_magnitude, self.m_thetaAngle)
        elif self.m_DFType == DF_BROADBAND:
            return self.broadband(self.m_halfLength, self.m_magnitude)
    
    def HFC(self, length, src):
        """Calculate the high-frequency content DF."""
        val = 0
        for i in range(length):
            val += src[i] * (i + 1)
        return val
    
    def specDiff(self, length, src):
        """Calculate the spectral difference DF."""
        val = 0.0
        for i in range(length):
            temp = abs((src[i] * src[i]) - (self.m_magHistory[i] * self.m_magHistory[i]))
            diff = math.sqrt(temp)
            val += diff
            self.m_magHistory[i] = src[i]
        return val


    def phaseDev(self, length, srcPhase):
        """Calculate the phase deviation DF."""
        val = 0
        for i in range(length):
            tmpPhase = (srcPhase[i] - 2 * self.m_phaseHistory[i] + self.m_phaseHistoryOld[i])
            dev = princarg(tmpPhase)
            tmpVal = abs(dev)
            val += tmpVal
            self.m_phaseHistoryOld[i] = self.m_phaseHistory[i]
            self.m_phaseHistory[i] = srcPhase[i]
        return val

    def complexSD(self, length, srcMagnitude, srcPhase):
        """Calculate the complex spectral difference DF."""
        val = 0
        for i in range(length):
            tmpPhase = (srcPhase[i] - 2 * self.m_phaseHistory[i] + self.m_phaseHistoryOld[i])
            dev = princarg(tmpPhase)
            meas = self.m_magHistory[i] - (srcMagnitude[i] * np.exp(1j * dev))
            tmpReal = np.real(meas)
            tmpImag = np.imag(meas)
            val += np.sqrt((tmpReal * tmpReal) + (tmpImag * tmpImag))
            self.m_phaseHistoryOld[i] = self.m_phaseHistory[i]
            self.m_phaseHistory[i] = srcPhase[i]
            self.m_magHistory[i] = srcMagnitude[i]
        return val
    
    def broadband(self, length, src):
        val = 0
        for i in range(length):
            sqrmag = src[i] * src[i]
            if self.m_magHistory[i] > 0.0:
                diff = 10.0 * math.log10(sqrmag / self.m_magHistory[i])
                if diff > self.m_dbRise:
                    val += 1
            self.m_magHistory[i] = sqrmag
        return val
    
    def princarg(ang):
        return mod(ang + np.pi, -2 * np.pi) + np.pi
    
    def mod(x, y):
        a = np.floor(x / y)
        b = x - (y * a)
        return b
    
    def getSpectrumMagnitude(self):
        return self.m_magnitude
    
        """
    A class for calculating detection functions from audio signals.

    The DetectionFunction class provides methods to compute various detection
    functions commonly used in music information retrieval tasks, such as tempo
    estimation and beat tracking. It supports different types of detection
    functions, including High-Frequency Content (HFC), spectral difference,
    phase deviation, complex spectral difference, and broadband detection.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration settings for the DetectionFunction object.
        The required keys are:
        - 'frameLength': The length of the analysis frame in samples.
        - 'stepSize': The step size between consecutive analysis frames in samples.
        - 'DFType': The type of detection function to compute.
        - 'dbRise': The required rise in decibels for the broadband detection function.
        - 'adaptiveWhitening': A boolean indicating whether to apply adaptive whitening.
        - 'whiteningRelaxCoeff': The relaxation coefficient for adaptive whitening.
        - 'whiteningFloor': The floor value for adaptive whitening.

    Attributes
    ----------
    m_dataLength : int
        The length of the analysis frame in samples.
    m_halfLength : int
        Half the length of the analysis frame plus one.
    m_DFType : int
        The type of detection function to compute.
    m_stepSize : int
        The step size between consecutive analysis frames in samples.
    m_dbRise : float
        The required rise in decibels for the broadband detection function.
    m_whiten : bool
        A boolean indicating whether to apply adaptive whitening.
    m_whitenRelaxCoeff : float
        The relaxation coefficient for adaptive whitening.
    m_whitenFloor : float
        The floor value for adaptive whitening.
    m_magHistory : numpy.ndarray
        The history of magnitude values for each frequency bin.
    m_phaseHistory : numpy.ndarray
        The history of phase values for each frequency bin.
    m_phaseHistoryOld : numpy.ndarray
        The old history of phase values for each frequency bin.
    m_magPeaks : numpy.ndarray
        The peak magnitude values for each frequency bin.

    Methods
    -------
    processTimeDomain(samples)
        Process the audio samples in the time domain and return the detection function value.
    processFrequencyDomain(reals, imags)
        Process the audio samples in the frequency domain and return the detection function value.
    whiten()
        Apply adaptive whitening to the magnitude spectrum.
    runDF()
        Run the selected detection function on the processed audio data.
    HFC(length, src)
        Calculate the High-Frequency Content (HFC) detection function.
    specDiff(length, src)
        Calculate the spectral difference detection function.
    phaseDev(length, srcPhase)
        Calculate the phase deviation detection function.
    complexSD(length, srcMagnitude, srcPhase)
        Calculate the complex spectral difference detection function.
    broadband(length, src)
        Calculate the broadband detection function.
    princarg(phase)
        Calculate the principal argument of the given phase.
    getSpectrumMagnitude()
        Return the magnitude spectrum of the processed audio signal.
    """