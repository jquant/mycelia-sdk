from base64 import b64decode, b64encode
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from PIL import Image
from tqdm import tqdm
import re

__all__ = ["read_image_folder"]


def encode_image(image) -> str:
    img_file = BytesIO()
    image.save(img_file, format="PNG")
    return b64encode(img_file.getvalue()).decode("utf-8")


def decode_image(encoded_string):
    return Image.open(BytesIO(b64decode(encoded_string))).convert("RGB")


def read_image_folder(image_folder: Union[Path, List[Path]],
                      resize: Tuple[int, int] = None,
                      handle_errors: str = "ignore",
                      id_pattern: str = None,
                      extensions: List = [".png", ".jpg", ".jpeg"]):
    """
    Function to read images from folder and transform to a format compatible
    to jai.

    Must pass the folder of the images or a list of paths for each image.

    Parameters
    ----------
    image_folder : str or Path, optional
        Path for the images folder. The default is None.
    new_size : Tuple of int, optional
        New shape to resize images. The default is None.
    handle_errors : str, optional
        Whether to ignore errors and skipped files. 
        If "ignore", could probably result in a internal error later on.
        The default is "ignore".
    id_pattern : str, optional
        regex string to find id value. The default is None.
    extensions : List, optional
        List of acceptable extensions.
        The default is [".png", ".jpg", ".jpeg"].

    Raises
    ------
    ValueError
        If not passed a image folder or list of images.

    Returns
    -------
    pd.Dataframe
        Pandas Dataframe with acceptable format for jai usage.

    """
    if isinstance(image_folder, (Path, str)):
        image_folder = Path(image_folder)
        name = image_folder.name
        loop_files = image_folder.iterdir()
    else:
        name = Path(image_folder[0]).name
        loop_files = chain(
            *[Path(folder).iterdir() for folder in image_folder])

    if handle_errors not in ['raise', 'warn', 'ignore']:
        raise ValueError("handle_errors must be 'raise', 'warn' or 'ignore'")

    encoded_images = []
    corrupted_files = []
    ignored_files = []
    for i, filename in enumerate(tqdm(loop_files)):
        if filename.suffix in extensions:
            if id_pattern is not None:
                i = int(re.search(id_pattern, filename.stem).group(1))

            try:
                im = Image.open(filename)
                im.verify()
                im.close()
                im = Image.open(filename)
                im.transpose(Image.FLIP_LEFT_RIGHT)
                im.close()

                img = Image.open(filename)
                if resize is not None:
                    img = img.resize(resize, Image.ANTIALIAS).convert('RGB')
                encoded_string = encode_image(img)

                # test if decoding is working
                decode_image(encoded_string)

            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as error:
                if handle_errors == 'raise':
                    raise ValueError(
                        f"file {filename} seems to be corrupted. {error}")

                corrupted_files.append(filename)
                continue

            encoded_images.append({
                'id': i,
                name: encoded_string,
                "filename": filename.name
            })
        else:
            if handle_errors == 'raise':
                raise ValueError(
                    f"file {filename} does not have the proper extension.\
                        acceptable extensions: {extensions}")
            ignored_files.append(filename)

    if handle_errors == 'warn' and len(ignored_files) > 0:
        print("Here are the ignored files:")
        ignored_message = '\n'.join(ignored_files)
        print(f"{ignored_message}")

    if handle_errors == 'warn' and len(corrupted_files) > 0:
        print("Here are the files that seem to be corrupted:")
        corrupted_message = '\n'.join(corrupted_files)
        print(f"{corrupted_message}")

    if len(encoded_images):
        return pd.DataFrame(encoded_images).set_index("id")
    return pd.DataFrame(encoded_images)