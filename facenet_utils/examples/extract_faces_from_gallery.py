#!/usr/bin/env python3
from absl import app, flags
from pathlib import Path
from facenet_utils.image_collection import ImageCollection
from facenet_utils.embedding import Embeddings
from facenet_utils.inference_wrappers import DetectionModel, EmbeddingModel

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '', 'Where to find models.')
flags.DEFINE_string('input_dir', '', 'File to output to.')
flags.DEFINE_string('debug_dir', '', 'File to output to.')


def main(argv):
  del argv    # Unused

  debug_dir = Path(FLAGS.debug_dir)
  debug_dir.mkdir(parents=True, exist_ok=True)
  
  print('Loading model')
  dmodel = DetectionModel(FLAGS.model_dir)
  emodel = EmbeddingModel(FLAGS.model_dir)

  ex = Embeddings('rj')

  cxn = ImageCollection.from_npz(FLAGS.input_dir)
  for img in cxn.images:
    new_crops = dmodel(img)
    ex.crops.extend(new_crops)
    for crop in new_crops:
      ex.embedding.append(emodel(crop))

  ex.to_file(f'{debug_dir}/vects.npz')


if __name__ == '__main__':
  app.run(main)
