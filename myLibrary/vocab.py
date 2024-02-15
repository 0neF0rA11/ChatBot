import pandas as pd

class Vocab:
    def __init__(self, corpus : pd.core.frame.DataFrame):
        corpus = corpus.copy()
        corpus['WordCount'] = corpus['Clear'].apply(lambda x: len(x.split()))
        self.max_seq_len = corpus['WordCount'].max()
        tokens = sorted(pd.Series(' '.join(corpus['Clear']).split()).unique())

        self.idx_to_token = {index: token for index, token in
                        enumerate(tokens, start=3)}
        self.idx_to_token[0] = "<PAD>"
        self.idx_to_token[1] = "<BOS>"
        self.idx_to_token[2] = "<EOS>"

        self.token_to_idx = {token: index for index, token in
                        enumerate(tokens, start=3)}
        self.token_to_idx["<PAD>"] = 0
        self.token_to_idx["<BOS>"] = 1
        self.token_to_idx["<EOS>"] = 2

        self.vocab_len = len(self.idx_to_token)