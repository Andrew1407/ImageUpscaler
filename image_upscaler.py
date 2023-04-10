from keras.engine.functional import Functional
import utils


class ImageUpscaler:
  def __init__(self, storage_path: str, model: Functional):
    self.__storage_path: str = storage_path
    self.__model: Functional = model


  def increase_resolution(self, image: bytes):
    prepared_data = utils.convert_raw_input(image)
    tensor = self.__model(prepared_data, training=False)
    upscaled_img = utils.unpack_tensor_image(tensor)
    saved = utils.save_as_file(self.__storage_path, upscaled_img)
    return saved
