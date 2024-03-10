import librosa

def load_audio_file(file_path):
    # Load the audio file using librosa
    audio_array, sample_rate = librosa.load(file_path, sr=None, mono=True)
    
    return audio_array, sample_rate