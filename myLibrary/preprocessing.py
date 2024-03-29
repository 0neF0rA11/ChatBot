import re
import pandas as pd

class DataProcessor:
    def __init__(self, data: pd.core.frame.DataFrame):
        """
        Класс для предварительной обработки данных

        Parameters:
        data (pd.DataFrame): Входной DataFrame, содержащий столбец текстовых данных.
        """
        self.data = data

    def preprocessing(self, text: str) -> str:
        """
        Выполняет предварительную обработку полученного текста

        Parameters:
        text (str): Текст для предварительной обработки

        Returns:
        str: Предварительно обработанный текст
        """
        text = text.lower()
        text = re.sub(r'[^а-я0-9 -]', '', text)
        text = text.replace(' - ', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def get_clear_data(self, column: str, add_new_column: bool = True) -> pd.core.frame.DataFrame:
        """
        Функция, предварительно обрабатывающая столбик с текстовыми данными

        Parameters:
        column (str): Имя столбца, содержащего текстовые данные, подлежащие предварительной обработке
        add_new_column (bool): Флаг добавления нового столбца
        
        Returns:
        pd.DataFrame: DataFrame с предварительно обработанными данными
        """
        self.clear_data = self.data.copy()

        if add_new_column:
            self.clear_data['Clear'] = self.clear_data[column].apply(self.preprocessing)
        else:
            self.clear_data[column] = self.clear_data[column].apply(self.preprocessing)

        return self.clear_data
    
    def write_clear_data(self, filename: str):
        """
        Функция, записывающая "чистые" данные в файл

        Parameters:
        filename (str): Имя файла, который будет записан в папке /date
        """
        self.clear_data.to_csv(f'data/{filename}', index=False)

