from torchvision.models import EfficientNet_V2_L_Weights
import plotly.express as px
import random
import torch
from torchvision import models
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


class EfficientNet_V2L(torch.nn.Module):
    transform = EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()
    name = 'efficientnet_v2_l'

    def __init__(self) -> None:
        """
        Initializes an EfficientNet model, cuts it at the global average pooling layer and returns the output features.
        """
        super().__init__()
        self.model = models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.IMAGENET1K_V1').eval()

    def forward(self, x) -> torch.tensor:
        x = self.model.features(x)
        x = self.model.avgpool(x)
        return x.squeeze(dim=3).squeeze(dim=2)

def imshow(img, rgb2bgr=True, title=None):
    img_show = img.copy()

    title = f"{img_show.shape}" if title is None else str(title)
    if rgb2bgr:
        img_show = img[..., ::-1]

    fig = px.imshow(img_show, binary_string=True, binary_format='png', title=title)
    fig.update_layout(
        showlegend=False,
        # width=1280, height=720,
        width=1080, height=720,
        autosize=False,
    )
    fig.show()

