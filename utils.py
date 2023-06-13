import os
from io import BufferedReader, BytesIO
import uuid
import tensorflow as tf
import numpy as np
from PIL import Image


RGB_MIN = 0
RGB_MAX = 255
RGB_MEAN = (RGB_MAX - RGB_MIN) / 2
NORMALIZATION_RANGE_LEN = 1

def convert_raw_input(image: bytes) -> np.ndarray:
  arr = np.array(Image.open(BytesIO(image)))
  normalized = (arr / RGB_MEAN) - NORMALIZATION_RANGE_LEN
  return np.expand_dims(normalized, axis=0) 


def unpack_tensor_image(eager_tensor: tf.TensorArray) -> Image.Image:
  clipped = tf.clip_by_value((eager_tensor + NORMALIZATION_RANGE_LEN) * RGB_MEAN, RGB_MIN, RGB_MAX)
  arr = clipped.numpy()[0]
  return Image.fromarray(arr.astype('uint8'), 'RGB')


def save_as_file(storage_path: str, image: Image.Image) -> BufferedReader:
  full_path = f'{storage_path}/{uuid.uuid1()}.png'
  image.save(full_path)
  file = open(full_path, 'rb')
  os.remove(full_path)
  return file
