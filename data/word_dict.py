import numpy as np


class WordDict():
    def __init__(self, embeddingPath):
        self.word2ID, self.ID2word, self.wordVector = self.loadEmbedding(embeddingPath)

    def loadEmbedding(self, Path):
        word2ID = {'UNK': 0}
        ID2word = {0: 'UNK'}
        wordVector = []

        with open(Path, encoding='utf-8') as f:
            idx = 1
            while True:
                line = f.readline()
                if not line:
                    break
                if idx == 1:
                    dim = len(line.split(' ')) - 1
                    unk_embedding = np.random.normal(0.0, 0.5, dim).astype('float32').tolist()
                    wordVector.append(unk_embedding)

                lineList = line.split(' ')
                word2ID[lineList[0]] = idx
                ID2word[idx] = lineList[0]
                wordVector.append([float(x) for x in lineList[1:]])

                idx += 1

        return word2ID, ID2word, wordVector

    def get_word2ID(self):
        return self.word2ID

    def get_ID2word(self):
        return self.ID2word

    def get_wordVector(self):
        return self.wordVector

if __name__ == '__main__':
    wordDict = WordDict('./glove.6B.100d.txt')
