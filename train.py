#!/usr/bin/env python3

import os, sys
from lang_classifier import LanguageClassifier
from analyzer import get_byte_n_grams

doc_directory = sys.argv[1]
classifier_output = sys.argv[2]

# initialize classifier with byte n-gram feature function (1 <= n <= 4)
byte_1_4_gram_analyzer = lambda x:get_byte_n_grams(x, n_gram_range=(1, 4))
classifier = LanguageClassifier(byte_1_4_gram_analyzer)

# get the list of folders in the training set
lang_folders = os.listdir(doc_directory)
assert(len(lang_folders)==21)

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
    classifier.partial_train(samples, labels)

