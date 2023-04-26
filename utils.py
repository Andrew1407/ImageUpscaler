import os
from io import BufferedReader, BytesIO
import uuid
import tensorflow as tf
import numpy as np
from PIL import Image


def convert_raw_input(image: bytes) -> np.ndarray:
  arr = np.array(Image.open(BytesIO(image)))
  normalized = (arr / 127.5) - 1
  return np.expand_dims(normalized, 0) 


def unpack_tensor_image(eager_tensor: tf.TensorArray) -> Image.Image:
  clipped = tf.clip_by_value((eager_tensor + 1) * 127.5, 0, 255)
  arr = clipped.numpy()[0]
  return Image.fromarray(arr.astype('uint8'), 'RGB')


def save_as_file(storage_path: str, image: Image.Image) -> BufferedReader:
  full_path = f'{storage_path}/{uuid.uuid1()}.png'
  image.save(full_path)
  file = open(full_path, 'rb')
  os.remove(full_path)
  return file
