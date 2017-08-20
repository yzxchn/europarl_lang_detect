#!/usr/bin/env python3

import os, sys
from lang_classifier import LanguageClassifier
from analyzer import byte_1_4_gram_analyzer
import pickle

doc_directory = sys.argv[1]
classifier_output = sys.argv[2]

classifier = LanguageClassifier(byte_1_4_gram_analyzer)

# get the list of folders in the training set
lang_folders = os.listdir(doc_directory)

for l in lang_folders:
    print("Training with files in folder {}".format(l))
    folder_path = os.path.join(doc_directory, l)
    files = os.listdir(folder_path)
    samples = []
    labels = []
    for f in files:
        file_path = os.path.join(folder_path, f)
        with open(file_path) as sample:
            samples.append(sample.read())
        labels.append(l)
    classifier.partial_train(samples, labels, lang_folders)

with open(classifier_output, 'wb') as out:
    pickle.dump(classifier, out)
