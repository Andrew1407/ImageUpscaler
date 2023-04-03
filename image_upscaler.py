import uuid
import os
from io import BufferedReader

import numpy as np
from PIL import Image
from io import BytesIO


class ImageUpscaler:
  def __init__(self, storage_path: str):
    self.__storage_path: str = storage_path


  def increase_resolution(self, image_bytes: bytes):
    image = self.__from_bytes_to_array(image_bytes)
    saved = self.__save_as_file(image)
    return saved


  def __from_bytes_to_array(self, bytes: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(bytes)))
  

  def __save_as_file(self, array: np.ndarray) -> BufferedReader:
    full_path = f'{self.__storage_path}/{uuid.uuid1()}.png'
    Image.fromarray(array).save(full_path)
    file = open(full_path, 'rb')
    os.remove(full_path)
    return file
