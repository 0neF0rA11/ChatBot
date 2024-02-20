import telebot
import torch
import json
import pickle
from myLibrary.myLSTM import MyLSTMModel


def load_config(config_file: str) -> dict:
    """
    Загружает конфигурацию модели из  файла

    Parameters:
    config_file (str): Путь до файла с конфигурацией модели (data/model_config.json)

    Returns:
    dict: Параметры для моей модели
    """
    with open(config_file, 'r') as json_file:
        params = json.load(json_file)
    return params

# Загрузка словаря токенов из pickle файла
with open('data/vocab.pkl', 'rb') as file:
    vocab = pickle.load(file)

# Инициализация и загрузка весов модели
model = MyLSTMModel(load_config('data/model_config.json'), vocab)
model.load_state_dict(torch.load('myLibrary/LSTM/best.pth', map_location=torch.device('cpu')))
model.eval()

bot = telebot.TeleBot('6869677738:AAFf5cJSosus2pDx3iVHLTnyTlUKxbwFuTQ')

@bot.message_handler(commands=['start'])
def send_welcome(message):
    """
    Обрабатывает команду "/start", отправляя приветственное сообщение.

    Parameters:
    message: объект сообщения Telegram
    """
    bot.send_message(message.chat.id, "Привет! Давай создавать? Ты начинай, а я продолжу: ")

@bot.message_handler(func=lambda message: True)
def generate_text(message):
    """
    Генерирует текст на основе сообщения пользователя и отправляет его в качестве ответа

    Parameters:
    message: объект сообщения Telegram
    """
    bot.send_message(message.chat.id, model.generate_text(message.text))

bot.polling()
