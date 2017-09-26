import os

import numpy as np
from sqlalchemy import (
    Column,
    Integer,
    String,
)
from sqlalchemy.ext.declarative import declarative_base
from sklearn.metrics.pairwise import cosine_similarity

from support import preprocess_audio
from model import MusicTaggerCRNN


class Comparator(object):

    def __init__(self, model):
        self.model = model

    def get_similarity(self, file_1, file_2):
        melgram_1 = preprocess_audio(file_1)
        melgrams_1 = np.expand_dims(melgram_1, axis=0)
        preds_1 = self.model.predict(melgrams_1)

        melgram_2 = preprocess_audio(file_2)
        melgrams_2 = np.expand_dims(melgram_2, axis=0)
        preds_2 = self.model.predict(melgrams_2)

        return cosine_similarity(preds_1, preds_2)


class MusicCorpus(object):

    class Item(declarative_base()):
        __tablename__ = "song_feature"
        file_path = Column(String(256), primary_key=True)
        deep_features = Column(String(512), index=True)

    def __init__(self, model, database):
        self.model = model

    def append(self, path):
        pass

    def get_nearest_neighbours(self, num, ):
        pass


if __name__ == '__main__':
    # for comparator test
    c = Comparator(MusicTaggerCRNN(include_top=False))
    # for file name
    s = c.get_similarity("data/mandy.mp3", "data/mandy.wma")
    print("=======================================")
    print("similarity:", s)
