#!/usr/bin/env python3

def get_byte_n_grams(text, n_gram_range=(1,1)):
    text_bytes = bytes(text, 'utf-8')
    ngrams=[]
    beg, end=n_gram_range
    for i in range(beg,end+1):
        ngrams.extend(get_n_grams(text_bytes, i))
    return map(to_string, ngrams)

def to_string(ngram):
    return " ".join(map(str, ngram))

def get_n_grams(seq, n):
    return zip(*[seq[i:] for i in range(n)])

def byte_1_4_gram_analyzer(x):
    return get_byte_n_grams(x, n_gram_range=(1,4))

def byte_unigram_analyzer(x):
    return get_byte_n_grams(x, n_gram_range=(1,1))
