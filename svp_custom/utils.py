import random
from tqdm import tqdm
from pathlib import Path

from typing import Union, List, Tuple


def glob_search(directories: Union[str, Path, List[str], List[Path]],
                pattern: str = '**/*',
                formats: Union[List[str], Tuple[str], str] = ('png', 'jpg', 'jpeg'),
                shuffle: bool = False,
                seed: int = 2,
                sort: bool = False,
                exception_if_empty=True,
                return_pbar=False) -> Union[List[Path], tqdm]:
    if isinstance(directories, (str, Path)):
        directories = [Path(directories)]
    files = []
    for directory in directories:
        if isinstance(directory, (str)):
            directory = Path(directory)
        if formats:
            if formats == '*':
                files.extend(directory.glob(f'{pattern}.{formats}'))
            else:
                for format in formats:
                    files.extend(directory.glob(f'{pattern}.{format.lower()}'))
                    files.extend(directory.glob(f'{pattern}.{format.upper()}'))
                    files.extend(directory.glob(f'{pattern}.{format.capitalize()}'))
        else:
            files.extend(directory.glob(f'{pattern}'))
    if exception_if_empty:
        if not len(files):
            raise Exception(f'There are no such files!')
    if shuffle:
        random.Random(seed).shuffle(files)
    if sort:
        files = sorted(files)
    if return_pbar:
        files = tqdm(files)
    return files
