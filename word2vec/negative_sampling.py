import random

def negative_sample(pos,vocabs):
    return random.sample(set(vocabs)-set(pos),len(pos))
