from ast import literal_eval
import shutil
import argparse
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from imagededup.methods import CNN, PHash
from imagededup.utils.general_utils import generate_files
from imagededup.utils.image_utils import load_image
from imagededup.utils.models import CustomModel
from svp_custom.utils import EfficientNet_V2L


def get_encoding_map(imgs: list[Path], encoder, bbox_csv: str | Path = None, skip_if_not_found=True) -> dict:
    result = {}
    if bbox_csv is not None:
        df = pd.read_csv(bbox_csv, index_col='path')
        df['bbox'] = df['bbox'].apply(lambda b: literal_eval(b))

    for img_path in tqdm(imgs, desc=f"encoding_map"):
        # get img array
        img = load_image(image_file=img_path)
        if bbox_csv is not None:
            try:
                xmin, ymin, xmax, ymax = df.loc[img_path.name]['bbox']
                img = img[ymin:ymax, xmin:xmax]
            except KeyError:
                if skip_if_not_found:
                    continue
        embedding = encoder.encode_image(image_array=img)[0]
        result[str(img_path)] = embedding
    return result


def main(args):
    if args.encoder == 'PHash':
        args.encoder = PHash()
    elif args.encoder == 'CNN':
        # args.encoder = CNN()
        args.encoder = CNN(model_config=CustomModel(
            model=EfficientNet_V2L(), transform=EfficientNet_V2L.transform, name=EfficientNet_V2L.name)
        )

    kwargs = {'num_enc_workers': args.workers,
              'min_similarity_threshold': args.thresh,
              #   'outfile': args.image_dir / 'tresh.json',
              }
    if args.bbox_csv:
        imgs = generate_files(args.image_dir, recursive=False)[::-1]
        # imgs = generate_files(args.image_dir, recursive=False)
        encoding_map = get_encoding_map(imgs, args.encoder, args.bbox_csv)
        kwargs['encoding_map'] = encoding_map
    else:
        kwargs['image_dir'] = args.image_dir

    duplicates = args.encoder.find_duplicates_to_remove(**kwargs)
    # duplicates = [args.image_dir / i for i in duplicates]

    shutil_f = shutil.move if args.mode == 'move' else shutil.copy

    for img_path in tqdm(duplicates, colour='green'):
        img_path = Path(img_path)

        new_img_path = args.image_dir / f'duplicate_{args.thresh}' / img_path.name
        new_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil_f(img_path, new_img_path)
        if args.suffix is not None:
            mark_path = img_path.with_suffix(f'.{args.suffix}')
            if mark_path.exists():
                new_mark_path = new_img_path.with_suffix(f'.{args.suffix}')
                shutil_f(mark_path, new_mark_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=str,
                        # required=True,
                        default='/home/spolyakov/Projects/common_processes/data/2246/stroi_full_by_camid/4cb7cd96-508b-409b-98e5-6c7a0a5fc139',
                        help='images directory source path (required)')
    # parser.add_argument('-t', '--bbox_csv', type=str, default=None, help='')
    parser.add_argument('-b', '--bbox_csv', type=str,
                        default='/home/spolyakov/Projects/common_processes/data/2246/df.csv', help='')

    parser.add_argument('-t', '--thresh', type=float, default=0.95, help='')  # 0.95
    parser.add_argument('-m', '--mode', choices=['copy', 'move'], default='move', help='')
    parser.add_argument('-e', '--encoder', choices=['CNN', 'PHash'], default='CNN', help='')
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
