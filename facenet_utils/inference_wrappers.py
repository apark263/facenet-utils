from typing import Optional, Tuple, List

import numpy as np
import PIL.Image
import tensorflow as tf
from fdlite import FaceDetection, FaceDetectionModel
from fdlite.types import Detection
from pathlib import Path
from facenet_utils.image_utils import prewhiten, crop_and_resize, TARGET_SIZE


class EmbeddingModel:
  def __init__(
      self,
      model_path: Optional[str] = None
  ) -> None:
    if not model_path:
      model_path = Path(__file__).parent.resolve().joinpath('models')
      model_path = model_path.joinpath('facenet512_quant.tflite')
    self.interpreter = tf.lite.Interpreter(model_path=f'{model_path}')
    self.interpreter.allocate_tensors()
    self.input_index = self.interpreter.get_input_details()[0]['index']
    self.input_shape = self.interpreter.get_input_details()[0]['shape']
    self.embedding_index = self.interpreter.get_output_details()[0]['index']

  def __call__(self, image: PIL.Image.Image) -> np.ndarray:
    batch = np.asarray(image, dtype=np.float32)[np.newaxis]
    self.interpreter.set_tensor(self.input_index, prewhiten(batch))
    self.interpreter.invoke()
    return self.interpreter.get_tensor(self.embedding_index)


class DetectionModel:
  """Wraps the FaceDetection class but can optionally returns crops
  instead of bounding boxes or a marked up image.

  Detections can also be filtered by score or size of bounding box.
  """
  def __init__(
      self,
      model_path: Optional[str] = None,
      min_size: int = 75,
      min_score: float = 0.9
  ) -> None:
    self.fd = FaceDetection(
        model_path=model_path,
        model_type=FaceDetectionModel.BACK_CAMERA)
    self.min_size = min_size
    self.min_score = min_score
    self.target_size = (TARGET_SIZE, TARGET_SIZE)

  def good(self, d: Detection) -> bool:
    if self.min_score and d.score < self.min_score:
      return False
    if self.min_size and d.bbox.width < self.min_size:
      return False
    return True

  def markup(self, img: PIL.Image.Image) -> PIL.Image.Image:
    detections = self(img)
    return img

  def crops(self, img: PIL.Image.Image) -> List[PIL.Image.Image]:
    detections = self(img)
    crops = []
    for d in detections:
      cropped = crop_and_resize(img, d, self.target_size)
      crops.append(cropped)
    return crops

  def __call__(self, img: PIL.Image.Image) -> List[Detection]:
    detections = self.fd(img)
    detections = [d.scaled(img.size) for d in detections]
    return [d.bbox.as_tuple for d in detections if self.good(d)]