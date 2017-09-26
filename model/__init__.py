try:
    from keras.applications.music_tagger_crnn import MusicTaggerCRNN
except ImportError:
    from .music_tagger_crnn import MusicTaggerCRNN


__all__ = [
    "MusicTaggerCRNN"
]
