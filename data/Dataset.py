from torch.utils.data import Dataset
import torch


class CoNLLDataset(Dataset):
    def __init__(self, dataPath, word2ID):
        super(CoNLLDataset, self).__init__()
        self.tags_id = {'I-LOC': 0, 'I-ORG': 1, 'B-MISC': 2, 'I-MISC': 3, 'B-ORG': 4, 'B-LOC': 5, 'O': 8, 'I-PER': 6,
                        'B-PER': 7, "<START>": 9, "<STOP>": 10}
        dataset, tags = self.readFromFile(dataPath)
        self.dataset = self.change_word_to_id(dataset, word2ID)
        self.tags = self.change_tag_to_id(tags)

    def readFromFile(self, dataPath):
        sentence_ls = []
        tag_ls = []

        with open(dataPath, encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip()
                lines.append(line)
        total_content = '\n'.join(lines)
        sentences = total_content.split('\n\n')

        for sentence in sentences:
            ls_tmp = sentence.strip().split('\n')
            tmp_sentence = []
            tmp_tag = []
            for ls in ls_tmp:
                tmp_sentence.append(ls.split(' ')[0].lower())
                tmp_tag.append(ls.split(' ')[-1])
            sentence_ls.append(tmp_sentence)
            tag_ls.append(tmp_tag)
        return sentence_ls, tag_ls
    def get_tag_index(self):
        return self.tags_id

    def change_word_to_id(self, dataset, word2ID):
        ls_tmp = []
        for ls in dataset:
            ls = [int(word2ID[x]) if word2ID.get(x) else int(word2ID['UNK']) for x in ls]
            ls_tmp.append(ls)
        return ls_tmp

    def change_tag_to_id(self, tags):
        ls_tmp = []
        for ls in tags:
            ls = [(self.tags_id[x]) if self.tags_id.get(x) else self.tags_id['O'] for x in ls]
            ls_tmp.append(ls)
        return ls_tmp

    def __getitem__(self, index):
        return torch.Tensor(self.dataset[index]).long(), torch.Tensor(self.tags[index]).long()

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from word_dict import WordDict
    from torch.utils.data import DataLoader

    wordDict = WordDict('./glove.6B.100d.txt')
    word2ID = wordDict.get_word2ID()
    dataset = CoNLLDataset('./CoNLL-2003/train.txt', word2ID)
    dataloader = DataLoader(dataset, shuffle=True)
    for i, data in enumerate(dataloader):
        print(data)
        break
