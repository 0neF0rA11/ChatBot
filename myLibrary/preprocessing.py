import re
import pandas as pd

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def preprocessing(self, text):
        text = text.lower()
        text = re.sub(r'[^а-я0-9 -]', '', text)
        text = text.replace(' - ', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def get_clear_data(self, column, add_new_column=True):
        self.clear_data = self.data.copy()
        
        if add_new_column:
            self.clear_data['Clear'] = self.clear_data[column].apply(self.preprocessing)
        else:
            self.clear_data[column] = self.clear_data[column].apply(self.preprocessing)

        return self.clear_data
    
    def write_clear_data(self, filename):
        self.clear_data.to_csv(f'data/{filename}', index=False)

