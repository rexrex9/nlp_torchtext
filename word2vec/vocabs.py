
def getvocabsOnlyIndex(seqs):
    vocabs = set()
    for seq in seqs:
        vocabs |= set(seq)
    vocabs = list(vocabs)
    return vocabs

def getVocabs(seqs):
    vacabs = set()
    for seq in seqs:
        vacabs |= set(seq)
    vacabs = list(vacabs)
    vacab_map = dict(zip(vacabs, range(len(vacabs))))
    return vacabs,vacab_map