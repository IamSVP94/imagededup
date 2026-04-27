import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from imagededup.methods import CNN, PHash
from imagededup.utils.models import CustomModel
from svp_custom.utils import EfficientNet_V2L


def main(args):
    if args.encoder == 'PHash':
        args.encoder = PHash()
    elif args.encoder == 'CNN':
        # args.encoder = CNN()
        args.encoder = CNN(model_config=CustomModel(
            model=EfficientNet_V2L(), transform=EfficientNet_V2L.transform, name=EfficientNet_V2L.name)
        )

    duplicates = args.encoder.find_duplicates_to_remove(
        image_dir=args.image_dir,
        num_enc_workers=args.workers,
        min_similarity_threshold=args.thresh,
        # outfile=args.image_dir / 'tresh.json',
    )

    duplicates = [args.image_dir / i for i in duplicates]
    shutil_f = shutil.move if args.mode == 'move' else shutil.copy

    for img_path in tqdm(duplicates, colour='green'):
        new_img_path = args.image_dir / f'duplicate_{args.thresh}' / img_path.name
        new_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil_f(img_path, new_img_path)
        if args.suffix is not None:
            mark_path = img_path.with_suffix(f'.{args.suffix}')
            if mark_path.exists():
                new_mark_path = new_img_path.with_suffix(f'.{args.suffix}')
                shutil_f(mark_path, new_mark_path)

        # plot_duplicates(image_dir=image_directory_path,
        #                 duplicate_map=duplicates,
        #                 filename='white_helmet_14_08_2023_0_00172_0_0_1.jpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=str,
                        # required=True,
                        default='/home/spolyakov/Downloads/other/special_types_cars/proj_spec_vehicles/sup23385/taxi',
                        help='images directory source path (required)')
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
