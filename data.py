from io import open
import unicodedata
import re

import torch
from torch.utils.data import Dataset

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
MAX_LENGTH = 20


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class MTDataset(Dataset):
    def __init__(
        self,
        part="train",
        input_lang=None,
        output_lang=None,
    ):
        src_lang = "eng"
        tgt_lang = "spa"
        if part == "train":
            self.input_lang, self.output_lang, self.pairs = prepareData(
                src_lang,
                tgt_lang,
                part,
            )
        elif part == "val":
            _, _, self.pairs = prepareData(src_lang, tgt_lang, part)
            self.input_lang = input_lang
            self.output_lang = output_lang
        elif part == "test":
            _, _, self.pairs = prepareData(src_lang, tgt_lang, part)
            self.input_lang = input_lang
            self.output_lang = output_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src = self.pairs[idx][0]
        tgt = self.pairs[idx][1]
        src_idxs = [
            self.input_lang.word2index[x]
            if x in self.input_lang.word2index.keys()
            else UNK_IDX
            for x in src
        ]
        tgt_idxs = [
            self.output_lang.word2index[x]
            if x in self.output_lang.word2index.keys()
            else UNK_IDX
            for x in tgt
        ]
        tgt_idxs = [SOS_IDX] + tgt_idxs + [EOS_IDX]
        return torch.tensor(src_idxs).long(), torch.tensor(tgt_idxs).long()


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, part, reverse=True):
    # print("Reading lines...")

    # Read the file and split into lines
    lines = (
        open("data/%s-%s-%s.txt" % (part, lang1, lang2), encoding="utf-8")
        .read()
        .strip()
        .split("\n")
    )

    if reverse:
        # Split every line into pairs and normalize
        pairs = [
            [normalizeString(s) for s in line.split("\t")][:2][::-1]
            for line in lines
        ]

        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        # Split every line into pairs and normalize
        pairs = [
            [normalizeString(s) for s in line.split("\t")][:2] for line in lines
        ]

        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return len(p[0]) < MAX_LENGTH and len(p[1]) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, part):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, part)
    # print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    # print("Trimmed to %s sentence pairs" % len(pairs))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs


def collate_samples(samples, padding_idx):
    batch_size = len(samples)
    max_seq_length_x = max([x.shape[0] for x, _ in samples])
    max_seq_length_y = max([y.shape[0] for _, y in samples])
    X_shape = (batch_size, max_seq_length_x)
    y_shape = (batch_size, max_seq_length_y)
    X = torch.zeros(X_shape, dtype=torch.long).fill_(padding_idx)
    Y = torch.zeros(y_shape, dtype=torch.long).fill_(padding_idx)
    for i, (x, y) in enumerate(samples):
        seq_len_x = x.shape[0]
        seq_len_y = y.shape[0]
        X[i, :seq_len_x] = x
        Y[i, :seq_len_y] = y
    return X, Y
