import telebot
import os
from dotenv import load_dotenv

from image_upscaler import ImageUpscaler
from upscaler_api import UpscalerApi


TEMP_DIR = 'temp'

if __name__ == '__main__':
  if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)
  load_dotenv()
  API_KEY = os.getenv('API_KEY')
  bot = telebot.TeleBot(API_KEY)
  image_upscaler = ImageUpscaler(TEMP_DIR)
  upscaler_bot = UpscalerApi(bot, image_upscaler)
  upscaler_bot.launch()
