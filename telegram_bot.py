import telebot
import json
from myLibrary.myLSTM import MyLSTMModel
import torch
import pickle

def load_config(config_file):
    with open(config_file, 'r') as json_file:
        params = json.load(json_file)
    return params

with open('data/vocab.pkl', 'rb') as file:
    vocab = pickle.load(file)

model = MyLSTMModel(load_config('data/model_config.json'), vocab)
model.load_state_dict(torch.load('LSTM/best.pth', map_location=torch.device('cpu')))
model.eval()

bot = telebot.TeleBot('<token>')

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Привет! Давай создавать? Ты начинай, а я продолжу: ")

@bot.message_handler(func=lambda message: True)
def generate_text(message):
    bot.send_message(message.chat.id, model.generate_text(message.text))

bot.polling()
