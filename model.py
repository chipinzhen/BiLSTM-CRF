from torch import nn
import torch


def log_sum_exp(vec):
    _, max_position = torch.max(vec, 1)
    max_position = max_position.item()
    max_score = vec[0, max_position]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BI_LSTM_CRF_MODEL(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tag_index, pre_trained_Embedding=None):
        super(BI_LSTM_CRF_MODEL, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_index = tag_index
        self.tagset_size = len(tag_index)

        if pre_trained_Embedding is not None:
            self.word_embeds = nn.Embedding.from_pretrained(pre_trained_Embedding, freeze=False)
        else:
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)  # fully connected network
        self.transitionMatrix = nn.parameter.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitionMatrix.data[:, tag_index["<START>"]] = -10000
        self.transitionMatrix.data[tag_index["<STOP>"], :] = -10000

        self.hidden = (torch.randn(2, 1, self.hidden_dim // 2),
                       torch.randn(2, 1, self.hidden_dim // 2))

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    # get the features from the lstm layer.
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_features = self.hidden2tag(lstm_out)
        return lstm_features

    def PathScore(self, feature_scores, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_index['<START>']], dtype=torch.long), tags])
        for i, feature_score in enumerate(feature_scores):
            score = score + feature_score[tags[i + 1]] + self.transitionMatrix[tags[i], tags[i + 1]]
        score = score + self.transitionMatrix[tags[-1], self.tag_index['<STOP>']]
        return score

    def Forward_algorithm(self, feature_scores):
        init_previous = torch.full((1, self.tagset_size), -10000)
        init_previous[0][self.tag_index['<START>']] = 0

        previous = init_previous

        for feature_score in feature_scores:
            forward_score = []

            for t in range(self.tagset_size):
                emit_score = feature_score[t].view(1, 1).expand(1, self.tagset_size)
                transition_score = self.transitionMatrix[:, t].view(1, self.tagset_size)
                next_tag_var = previous + transition_score + emit_score
                forward_score.append(log_sum_exp(next_tag_var).view(1))
            previous = torch.cat(forward_score).view(1, -1)
        final_score = previous + self.transitionMatrix[:, self.tag_index['<STOP>']].view(1, self.tagset_size)
        final_score = log_sum_exp(final_score)
        return final_score

    def forward(self, sentence, tags):
        features_scores = self._get_lstm_features(sentence)
        forward_score = self.Forward_algorithm(features_scores)
        path_score = self.PathScore(features_scores, tags)
        # print(forward_score, path_score)
        return forward_score - path_score
