import shutil

import faiss
import numpy as np
from pathlib import Path
from imagededup.methods import CNN
from svp_custom.utils import glob_search

src_dir = '/home/vid/hdd/file/project/263-КоникаМинольта/5-Danon/img/ATTRS/ATTRS_v3_LABELME/ATTRS_v3_LABELME-cup_poured_out_jet_2007_to_s003/'
dst_dir = '/home/vid/hdd/file/project/263-КоникаМинольта/5-Danon/img/ATTRS/ATTRS_v3_LABELME/train-cup_poured_out_to_s003/'
annot_suffix = '.json'
src_dir = Path(src_dir)
new_src_dir = src_dir.parent / f"{src_dir.name}_renamed"


def get_embeddings(dir_path, encoder=CNN):
    current_encoder = encoder()
    filenames, embeddings = [], []
    imgs = glob_search(dir_path, return_pbar=True)
    for img_path in imgs:
        filenames.append(img_path)
        embedding = current_encoder.encode_image(img_path)
        embeddings.append(embedding)
    else:
        embeddings = np.squeeze(np.array(embeddings))
    return filenames, embeddings


src_filenames, src_embeddings = get_embeddings(src_dir)
dst_filenames, dst_embeddings = get_embeddings(dst_dir)

assert src_embeddings.shape[-1] == dst_embeddings.shape[-1], (
    f"Embeddings have different dimensions: {src_embeddings.shape[-1]} != {dst_embeddings.shape[-1]}"
)

dim = src_embeddings.shape[-1]
index = faiss.IndexFlatL2(dim)
index.add(dst_embeddings)

topn = 1
D, I = index.search(src_embeddings, topn)

for src_img_path, dst_idxs in zip(src_filenames, I):
    dst_img_path = dst_filenames[dst_idxs[0]]

    new_img_path = new_src_dir / dst_img_path.name

    counter = 1
    while new_img_path.exists():
        new_img_path = new_src_dir / f'{dst_img_path.stem}__[{counter}]{dst_img_path.suffix}'
        counter += 1

    new_img_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_img_path, new_img_path)  # copy img

    annot_path = src_img_path.with_suffix(annot_suffix)
    if annot_path.exists():
        new_annot_path = new_img_path.with_suffix(annot_suffix)
        shutil.copy(annot_path, new_annot_path)  # copy annot
