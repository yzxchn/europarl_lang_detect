#!/usr/bin/env python3
"""Serves as a wrapper for MultinomialNB classifier with feature vectorizer
included.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class LanguageClassifier:
    def __init__(self, feature_function):
        self.vectorizer = CountVectorizer(analyzer=feature_function)
        self.classifier = MultinomialNB()

    def partial_train(samples, labels):
        """Partially train the algorithm, using MultinomialNB's partial_fit 
        method. Use this when the size of the training sample set is too big 
        and training on multiple batches is required.
        """
        sample_vectors = self.vectorizer(samples)
        self.classifier.partial_fit(sample_vectors, labels)

    def train(samples, labels):
        sample_vectors = self.vectorizer(samples)
        self.classifier.fit(sample_vectors, labels)

    def predict(samples):
        sample_vectors = self.vectorizer(samples)
        return self.classifier.predict(sample_vectors)
