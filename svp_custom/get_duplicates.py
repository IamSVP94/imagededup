import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from imagededup.methods import CNN, PHash
from svp_custom.utils import glob_search


def main(args):
    if args.encoder == 'PHash':
        args.encoder = PHash()
    elif args.encoder == 'CNN':
        args.encoder = CNN()

    embeddings = []
    imgs = glob_search(args.image_dir)[:100]
    pbar = tqdm(imgs)
    for img_path in pbar:
        pbar.set_description(str(img_path))
        embedding = args.encoder.encode_image(img_path)
        embeddings.append(embedding)
    etalonImgEmbedding = args.encoder.encode_image(args.image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str,
                        # required=True,
                        default="/home/vid/hdd/datasets/VLADISLAV/dsml/example.png",
                        help='images directory source path (required)')
    parser.add_argument('-d', '--image_dir', type=str,
                        default="/home/vid/hdd/datasets/VLADISLAV/dsml/vmx_gazovanie/NLMK_gaz_door_labelme/",
                        help='images directory source path (required)')
    parser.add_argument('-e', '--encoder', choices=['CNN', 'PHash'], default='CNN', help='')
    parser.add_argument('-t', '--thresh', type=float, default=0.995, help='')
    parser.add_argument('-m', '--mode', choices=['copy', 'move'], default='move', help='')
    parser.add_argument('-s', '--suffix', choices=['txt', 'json', '-'], default='-', help='')
    parser.add_argument('-w', '--workers', type=int, default=15, help='')
    args = parser.parse_args()

    print()
    print(args.image_dir)
    print()

    args.image_dir = Path(args.image_dir)
    if args.suffix == '-':
        args.suffix = None
    main(args)
