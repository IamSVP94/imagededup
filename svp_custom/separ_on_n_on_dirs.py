import argparse
import shutil
from pathlib import Path

from tqdm import trange, tqdm


def main(args):
    shutil_f = shutil.move if args.mode == 'move' else shutil.copy
    files = [x for x in args.src_dir.glob('**/*') if x.is_file()]
    counter = 0
    l = 0
    for r in trange(args.split_number, len(files) + args.split_number, args.split_number):
        r = min(r, len(files) + 1)
        current_files = files[l:r]
        dst_dir = args.dst_dir / str(counter)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f_path in tqdm(current_files, leave=False):
            new_f_path = dst_dir / f_path.name
            shutil_f(str(f_path), str(new_f_path))
        counter += 1
        l = r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src_dir', type=str,
                        # required=True,
                        default='/home/vid/hdd/file/project/263-КоникаМинольта/5-Danon/video/0611/imgs/6230_func_test_to_s003_from_s003_detobj/person/123_Водитель_взял_пробоотборник_из_ванны_20240628/1_2024-06-28_04-46-33_574.mkv',
                        help='')
    parser.add_argument('-d', '--dst_dir', type=str, default=None, help='')
    parser.add_argument('-n', '--split_number', type=int, default=9990, help='')
    parser.add_argument('-m', '--mode', choices=['copy', 'move'], default='move', help='')

    args = parser.parse_args()

    args.src_dir = Path(args.src_dir).resolve()
    args.dst_dir = Path(
        args.dst_dir).resolve() if args.dst_dir else args.src_dir.parent / f'{args.src_dir.name}_dirs'

    main(args)
