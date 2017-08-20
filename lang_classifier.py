#!/usr/bin/env python3
"""Serves as a wrapper for MultinomialNB classifier with feature vectorizer
included.
"""

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

class LanguageClassifier:
    def __init__(self, feature_function):
        self.vectorizer = HashingVectorizer(analyzer=feature_function, 
                                            norm=None, 
                                            non_negative='total')
        self.classifier = MultinomialNB()

    def partial_train(self, samples, labels, classes):
        """Partially train the algorithm, using MultinomialNB's partial_fit 
        method. Use this when the size of the training sample set is too big 
        and training on multiple batches is required.
        """
        sample_vectors = self.vectorizer.fit_transform(samples)
        self.classifier.partial_fit(sample_vectors, labels, classes)

    def train(self, samples, labels):
        sample_vectors = self.vectorizer.fit_transform(samples)
        self.classifier.fit(sample_vectors, labels)

    def predict(self, samples):
        sample_vectors = self.vectorizer.transform(samples)
        return self.classifier.predict(sample_vectors)

