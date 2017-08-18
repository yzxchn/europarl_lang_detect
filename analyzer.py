#!/usr/bin/env python3

def get_byte_n_grams(text, n_gram_range=(1,1)):
    text_bytes = bytes(text, 'utf-8')
    ngrams=[]
    beg, end=n_gram_range
    for i in range(beg,end+1):
        ngrams.extend(get_n_grams(text_bytes, i))
    return ngrams

def get_n_grams(seq, n):
    return zip(*[seq[i:] for i in range(n)])
