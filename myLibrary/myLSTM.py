import re
import torch
import torch.nn as nn


class MyLSTMModel(nn.Module):
  def __init__(self, params, vocab):
    super(MyLSTMModel, self).__init__()
    self.vocab = vocab
    self.embedding = nn.Embedding(params['input_size'], embedding_dim=params['embedding_dim'])
    self.lstm = nn.LSTM(params['embedding_dim'], params['hidden_size'], params['num_layers'], batch_first=True,
                        bidirectional=True, dropout=params['dropout_p'])
    self.fc = nn.Linear(params['hidden_size'] * 2, params['output_size'])

  def forward(self, x):
    embedded = self.embedding(x)
    out, _ = self.lstm(embedded)
    out = self.fc(out[:, -1, :])
    return out
  
  def preprocessing(self, text):
    text = text.lower()
    text = re.sub(r'[^а-я0-9 -]', '', text)
    text = text.replace(' - ', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

  def generate_text(self, string):
    string = f'<BOS> {self.preprocessing(string)}'
    tokens = string.split()
    list_of_tokens = [self.vocab.token_to_idx.get(word, 0) for word in tokens]
    list_of_tokens = (self.vocab.max_seq_len -len(list_of_tokens)) * [0] + list_of_tokens
    random_sent = torch.tensor(list_of_tokens)
    
    i = 3
    while i != 0:
        pred = self.forward(random_sent.view(1, -1))
        pred_token = pred.argmax()
        if pred_token == 2:
          i -= 1
        list_of_tokens.append(pred_token.item())
        string += f' {self.vocab.idx_to_token[pred_token.item()]}'
        random_sent = torch.tensor(list_of_tokens[-len(random_sent):])

    string = string.replace(' <EOS>', '.')
    string = string.replace('<BOS> ', '')

    sentences = string.split('.')
    sentences = [sentence.strip().capitalize() for sentence in sentences if sentence]
    punct_string = '. '.join(sentences) + '.'
    return punct_string
