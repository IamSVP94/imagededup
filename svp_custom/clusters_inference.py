import shutil
import numpy as np
from tqdm import tqdm
from imagededup.methods import CNN
from sklearn.cluster import KMeans
from svp_custom.utils import glob_search

dir = '/home/vid/hdd/file/project/263-КоникаМинольта/5-Danon/video/0611/imgs/6230_func_test_to_s003_from_s003_detobj/person/510_Установка_башмаков_20240730/1_2024-07-30_13-18-07_4621.mkv'
n_clusters = 3
mode = 'move'

imgs = glob_search(dir)
shutil_f = shutil.move if mode == 'move' else shutil.copy

cnn_encoder = CNN()
embeddings = []
pbar = tqdm(imgs)
for img_path in pbar:
    pbar.set_description(str(img_path))
    embedding = cnn_encoder.encode_image(img_path)
    embeddings.append(embedding)
else:
    embeddings = np.squeeze(np.array(embeddings))

kmeans = KMeans(n_clusters=n_clusters, verbose=1)
kmeans.fit(embeddings)

labels = kmeans.labels_
for img_path, label in zip(imgs, labels):
    new_path = img_path.parent / str(label) / img_path.name
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil_f(str(img_path), str(new_path))
