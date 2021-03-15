from pathlib import Path
import sys

jai_folder = Path.cwd().parent.parent  # no one's proud of this
sys.path.append(jai_folder.as_posix())

from jai.image import read_image_folder


def generate_read_image_folder(image_folder=Path("test_imgs")):
    img_data = read_image_folder(image_folder=image_folder)
    print()
    print(img_data)
    img_data.to_pickle(Path("test_imgs/dataframe_img.pkl"))


if __name__ == '__main__':
    generate_read_image_folder()
