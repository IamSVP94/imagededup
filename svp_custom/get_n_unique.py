import shutil
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from imagededup.methods import CNN
from imagededup.utils.models import CustomModel
from svp_custom.utils import EfficientNet_V2L

img_dir = '/home/spolyakov/Downloads/other/special_types_cars/proj_spec_vehicles/sup23385/rescue_service/unique/by_cam/c5a06b6e-15c7-48fe-8f33-0982ea48bd14'
n = 20
mode = 'move'

# ----------------------------------------------------------

img_dir = Path(img_dir)
shutil_f = shutil.move if mode == 'move' else shutil.copy

# cnn_encoder = CNN()
cnn_encoder = CNN(model_config=CustomModel(model=EfficientNet_V2L(),
                                           transform=EfficientNet_V2L.transform,
                                           name=EfficientNet_V2L.name))


# get embeddings
path_embs_dict = cnn_encoder.encode_images(img_dir, num_enc_workers=15)
embeddings = np.array(list(path_embs_dict.values()))
faiss.normalize_L2(embeddings)

dim = embeddings.shape[-1]

# get centers
kmeans = faiss.Kmeans(dim, n, gpu=True, niter=20, nredo=5, verbose=True)
kmeans.train(embeddings)
centers = kmeans.centroids

# get nearest to centers
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

D, I = index.search(centers, 1)  # 1 nearest neighbor per center
selected_indices = I.flatten()

# separate nearest to centers from another
new_dir = img_dir / f'unique_{n}'
new_dir.mkdir(parents=True, exist_ok=False)
for idx, img_path in enumerate(tqdm(path_embs_dict.keys())):
    if idx in selected_indices:
        img_path = img_dir / img_path
        new_path = new_dir / img_path.name
        shutil_f(str(img_path), str(new_path))
