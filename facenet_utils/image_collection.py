import io
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import PIL.Image


_IMG_SUFFIXES = ['.jpg', '.gif', '.png', '.jpeg']


def _get_img_files(input_dir: str):
  return [p.resolve() for p in Path(input_dir).glob("**/*")
          if p.suffix.lower() in _IMG_SUFFIXES]


def _get_duration(filename: str) -> float:
  sub_args = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
              '-of', 'default=noprint_wrappers=1:nokey=1', filename,
              ]
  result = subprocess.run(sub_args, stdout=subprocess.PIPE)
  return float(result.stdout.decode())


def _get_frame(filename: str, timestamp: float) -> PIL.Image.Image:
  sub_args = ['ffmpeg', '-v', 'error', '-ss', f'{timestamp}', '-i', filename,
              '-vframes', '1', '-c:v', 'png', '-f', 'rawvideo', '-',
              ]
  result = subprocess.run(sub_args, stdout=subprocess.PIPE)
  return PIL.Image.open(io.BytesIO(result.stdout))


def _get_frames(filename: str, num_frames: int = 30) -> List[PIL.Image.Image]:
  duration = _get_duration(filename)
  chunk = duration / (num_frames + 1)
  return [_get_frame(filename, i*chunk) for i in range(1, num_frames + 1)]


@dataclass
class ImageCollection:
  images: List[PIL.Image.Image] = field(default_factory=list)

  @classmethod
  def from_directory(cls, directory: str):
    files = _get_img_files(directory)
    images = [PIL.Image.open(f) for f in files]
    return cls(images=images)

  @classmethod
  def from_npz(cls, file: str):
    x = np.load(file)
    images = [PIL.Image.open(io.BytesIO(b)) for b in x['blobs']]
    return cls(images=images)

  @classmethod
  def from_vid(cls, file: str, num_frames: int = 30):
    images = _get_frames(file, num_frames)
    return cls(images=images)
