import telebot
import os
from dotenv import load_dotenv
import tensorflow as tf

from image_upscaler import ImageUpscaler
from upscaler_api import UpscalerApi


TEMP_DIR = 'temp'
MODEL_PATH = 'model/upscaler'


def make_upscaler(token: str) -> UpscalerApi:
  bot = telebot.TeleBot(token)
  model = tf.keras.models.load_model(MODEL_PATH, compile=True)
  image_upscaler = ImageUpscaler(TEMP_DIR, model)
  return UpscalerApi(bot, image_upscaler)


if __name__ == '__main__':
  if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)
  load_dotenv()
  token = os.getenv('API_KEY')
  upscaler = make_upscaler(token)
  upscaler.launch()
