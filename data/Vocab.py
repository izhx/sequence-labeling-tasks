import numpy as np


class Vocab(object):
    PAD, ROOT, UNK = 0, 1, 2

    def __init__(self, word_counter, rel_counter, relroot='root', min_occur_count=2):
        self._root = relroot
        self._root_form = '<' + relroot.lower() + '>'
        self._id2word = ['<pad>', self._root_form, '<unk>']
        self._wordid2freq = [10000, 10000, 10000]
        self._id2rel = ['<pad>', relroot]
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)
                self._wordid2freq.append(count)

        for rel, count in rel_counter.most_common():
            if rel != relroot:
                self._id2rel.append(rel)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        self._rel2id = reverse(self._id2rel)
        if len(self._rel2id) != len(self._id2rel):
            print("serious bug: relation labels dumplicated, please check!")

        print("Vocab info: #words %d, #rels %d" % (self.vocab_size, self.rel_size))

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        self._id2extword = []
        allwords = set()
        for special_word in ['<pad>', self._root_form, '<unk>']:
            if special_word not in allwords:
                allwords.add(special_word)
                self._id2extword.append(special_word)

        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) > 10:
                    curword = values[0]
                    if curword not in allwords:
                        allwords.add(curword)
                        self._id2extword.append(curword)
                    embedding_dim = len(values) - 1
        word_num = len(self._id2extword)
        print('Total words: ' + str(word_num) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: words dumplicated, please check!")

        oov_id = self._extword2id.get('<unk>')
        if self.UNK != oov_id:
            print("serious bug: oov word id is not correct, please check!")

        embeddings = np.zeros((word_num, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) > 10:
                    index = self._extword2id.get(values[0])
                    vector = np.array(values[1:], dtype='float64')
                    embeddings[index] = vector
                    embeddings[self.UNK] += vector
        embeddings[self.UNK] = embeddings[self.UNK] / word_num
        embeddings = embeddings / np.std(embeddings)
        return embeddings

    def create_placeholder_embs(self, embfile):
        word_num = len(self._id2extword)
        embedding_dim = -1
        embeddings = None
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) > 10:
                    if embedding_dim == -1:
                        embedding_dim = len(values) - 1
                        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')
                        embeddings = np.zeros((word_num, embedding_dim))
                    index = self._extword2id.get(values[0])
                    vector = np.array(values[1:], dtype='float64')
                    embeddings[index] = vector
                    embeddings[self.UNK] += vector
        if embeddings is not None:
            embeddings[self.UNK] = embeddings[self.UNK] / word_num
            embeddings = embeddings / np.std(embeddings)
        return embeddings

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def wordid2freq(self, xs):
        if isinstance(xs, list):
            return [self._wordid2freq[x] for x in xs]
        return self._wordid2freq[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.UNK) for x in xs]
        return self._extword2id.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self._id2extword[x] for x in xs]
        return self._id2extword[xs]

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id.get(x, self.ROOT) for x in xs]
        return self._rel2id.get(xs, self.ROOT)

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def extvocab_size(self):
        return len(self._id2extword)

    @property
    def rel_size(self):
        return len(self._id2rel)

