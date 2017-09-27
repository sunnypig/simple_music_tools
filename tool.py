import os

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors.classification import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier
)

from support import preprocess_audio
from model import MusicTaggerCRNN


class Parser(object):

    def __init__(self, model):
        self.model = model

    def get_features(self, file):
        melgram = preprocess_audio(file)
        melgrams = np.expand_dims(melgram, axis=0)
        features = self.model.predict(melgrams)
        return features.reshape(-1)


class Comparator(Parser):

    def get_similarity(self, file_1, file_2):
        return cosine_similarity(
            self.get_features(file_1),
            self.get_features(file_2)
        )


class MusicCorpus(Parser):

    def __init__(self, model):
        super().__init__(model)
        self.corpus = pd.DataFrame(
            columns=["path", "features"])
        self.classifier = RadiusNeighborsClassifier()

    def scan(self, path):
        path = os.path.abspath(path)
        len_orig = len(self.corpus)
        for dir_name, _, file_names in os.walk(path):
            for file_name in file_names:
                file_path = os.path.join(dir_name, file_name)
                print("parsing file:", file_path)
                self.corpus.loc[len(self.corpus)] = [
                    file_path, self.get_features(file_path)]

        len_inc = len(self.corpus) - len_orig
        if len_inc:
            print("fitting to classifier with", len_inc, "files")
            self.classifier.fit(
                self.corpus["features"][len_orig:].tolist(),
                np.arange(len(self.corpus))
            )

    def find_similar_audio(self, radius=2.0):
        similar_group = []
        for features in self.corpus["features"]:
            neighbours, = self.classifier.radius_neighbors(
                [features,], radius=radius, return_distance=False)
            if len(neighbours) > 1:
                similar_group.append(
                    [self.corpus["path"][n] for n in neighbours])
        return similar_group


if __name__ == '__main__':
    # for comparator test
    m = MusicTaggerCRNN(include_top=False)
    # c = Comparator(m)
    # # for file name
    # s = c.get_similarity("data/mandy.mp3", "data/mandy.wma")
    # print("=======================================")
    # print("similarity:", s)

    # for corpus test
    c = MusicCorpus(m)
    c.scan("data")
    print("=======================================")
    print(c.find_similar_audio(radius=1.0))
