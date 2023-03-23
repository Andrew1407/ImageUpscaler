import uuid
import os
from PIL import Image


class ImageUpscaler:
  def __init__(self, storage_path: str):
    self.__storage_path: str = storage_path


  def increase_resolution(self, image):
    full_path = f'{self.__storage_path}/{uuid.uuid1()}.png'
    Image.fromarray(image).save(full_path)
    file = open(full_path, 'rb')
    os.remove(full_path)
    return file
