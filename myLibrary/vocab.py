import pandas as pd

class Vocab:
    def __init__(self, corpus: pd.core.frame.DataFrame):
        """
        Генерирует словарь для заданного корпуса
        
        Args:
            corpus (pd.core.frame.DataFrame): DataFrame со столбцом "Clear", содержащим текстовые данные
            
        Attributes:
            max_seq_len (int): Максимальное количество слов в сэмпле
            idx_to_token (dict): Словарь перевода из индекса в токен
            token_to_idx (dict): Словарь перевода из токена в индекс
            vocab_len (int): Длина словаря 
        """
        corpus = corpus.copy()
        corpus['WordCount'] = corpus['Clear'].apply(lambda x: len(x.split()))
        self.max_seq_len = corpus['WordCount'].max()
        tokens = sorted(pd.Series(' '.join(corpus['Clear']).split()).unique())

        self.idx_to_token = {index: token for index, token in enumerate(tokens, start=3)}
        self.idx_to_token[0] = "<PAD>"
        self.idx_to_token[1] = "<BOS>"
        self.idx_to_token[2] = "<EOS>"

        self.token_to_idx = {token: index for index, token in self.idx_to_token.items()}

        self.vocab_len = len(self.idx_to_token)