from keras.engine.functional import Functional
import utils


class ImageException(Exception): ...


class ImageUpscaler:
  def __init__(self, storage_path: str, model: Functional):
    self.__storage_path: str = storage_path
    self.__model: Functional = model
    self.__max_image_size: tuple[int, int] = (400, 400)


  def increase_resolution(self, image: bytes):
    prepared_data = utils.convert_raw_input(image)
    self.__check_image_size(prepared_data.shape[1:3])
    tensor = self.__model(prepared_data, training=False)
    upscaled_img = utils.unpack_tensor_image(tensor)
    saved = utils.save_as_file(self.__storage_path, upscaled_img)
    return saved

  
  def __check_image_size(self, shape: tuple):
    width, height = shape
    if width > self.__max_image_size[0] or width > self.__max_image_size[1]:
      message = f'{(width, height)} size is too big to upscale. Please pass images smaller than or equal {(self.__max_image_size)}.'
      raise ImageException(message)
