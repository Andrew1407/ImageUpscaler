import telebot
import os
from dotenv import load_dotenv
import tensorflow as tf

from image_upscaler import ImageUpscaler
from upscaler_api import UpscalerApi


TEMP_DIR = 'temp'
MODEL_PATH = 'model/upscaler'


if __name__ == '__main__':
  if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)
  
  load_dotenv()
  API_KEY = os.getenv('API_KEY')

  # tf.keras.models.load_model(MODEL_PATH)

  bot = telebot.TeleBot(API_KEY)
  image_upscaler = ImageUpscaler(TEMP_DIR)
  upscaler = UpscalerApi(bot, image_upscaler)

  upscaler.launch()
