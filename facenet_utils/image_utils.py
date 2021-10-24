import io
import numpy as np
import PIL.Image
from typing import List, Tuple, Optional

TARGET_SIZE = 160
TILE_LAYOUT = 6


def prewhiten(x: np.ndarray) -> np.ndarray:
  assert x.ndim == 4, 'Should represent a tensor of rank 4.'
  axes = (1, 2, 3)
  mean = np.mean(x, axis=axes, keepdims=True)
  std = np.std(x, axis=axes, keepdims=True)
  std_adj = np.maximum(std, 1.0/np.sqrt(x[0].size))
  y = (x - mean) / std_adj
  return y


def crop_and_resize(img: PIL.Image.Image,
                    roi: Tuple[int, int, int, int],
                    output_size: Optional[Tuple[int, int]] = None) -> PIL.Image.Image:
  if output_size is None:
    output_size = (roi[2] - roi[0], roi[3] - roi[1])
  output = img.transform(size=output_size, method=PIL.Image.EXTENT,
                         data=roi, resample=PIL.Image.BICUBIC)
  return output


def tile_images(image_list: List[PIL.Image.Image]) -> PIL.Image.Image:
  cols = TILE_LAYOUT
  rows = - (- len(image_list) // cols)
  dst = PIL.Image.new('RGB', (cols * TARGET_SIZE, rows * TARGET_SIZE))
  for idx, img in enumerate(image_list):
    row, col = idx // cols, idx % cols
    dst.paste(img, (col * TARGET_SIZE, row * TARGET_SIZE))
  return dst


def img_to_bytes(img: PIL.Image.Image) -> bytes:
  with io.BytesIO() as output:
    img.save(output, format='JPEG', quality='high')
    return output.getvalue()


def bytes_to_img(blob: bytes) -> PIL.Image.Image:
  return PIL.Image.open(io.BytesIO(blob))