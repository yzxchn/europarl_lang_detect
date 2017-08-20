#!/usr/bin/env python3

import pickle
import sys, os
from sklearn.metrics import classification_report, accuracy_score

test_path = sys.argv[1]
classifier_path = sys.argv[2]

with open(classifier_path, 'rb') as cls_in:
    classifier = pickle.load(cls_in)

# load test samples
true_labels = []
sents = []
with open(test_path) as test:
    for l in test:
        lang, sent = l.strip().split('\t')
        true_labels.append(lang)
        sents.append(sent)

predicted = classifier.predict(sents)

report = classification_report(true_labels, predicted, digits=4)
accuracy = accuracy_score(true_labels, predicted)

print(report)

print("Accuracy: {}".format(accuracy))
