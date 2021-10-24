from dataclasses import dataclass, field
from typing import List

import numpy as np
import PIL.Image

from facenet_utils.image_utils import bytes_to_img, img_to_bytes


@dataclass
class Embeddings:
  key: str
  embedding: List[np.ndarray] = field(default_factory=list)
  crops: List[PIL.Image.Image] = field(default_factory=list)

  @property
  def value(self):
    return np.mean(self.embedmat, axis=0)

  def distance(self, other):
    return np.linalg.norm(self.value - other)

  @property
  def blobs(self):
    return [img_to_bytes(i) for i in self.crops]

  @property
  def embedmat(self):
    return np.stack(self.embedding, axis=0)

  @property
  def distmat(self):
    return np.linalg.norm(self.embedmat - self.embedmat.transpose(1, 0, 2), axis=-1)

  @classmethod
  def from_file(cls, model_file: str):
    x = np.load(model_file)
    e = x['embedmat']
    embedding = np.split(e, len(e))
    crops = [bytes_to_img(b) for b in x['blobs']]
    return cls(x['key'], embedding, crops)

  def to_file(self, model_file: str):
    np.savez(model_file, key=self.key,
             embedmat=self.embedmat, blobs=self.blobs, tiled_img=self.tiled_img)

  def prune_rows(self, threshold):
    idx = np.where(self.distmat.mean(axis=0) < threshold)[0]
    new_embed = [self.embedding[i] for i in idx]
    new_crops = [self.crops[i] for i in idx]
    self.embedding = new_embed
    self.crops = new_crops
