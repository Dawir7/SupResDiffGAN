import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def copy_file(src, dst):
    shutil.copy(src, dst)


def copy_files(file_list, src_dir, dst_dir, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                copy_file, os.path.join(src_dir, f), os.path.join(dst_dir, f)
            )
            for f in file_list
        ]
        for future in futures:
            future.result()


def resize_and_save_image(file, src_dir, dest_dir, scale):
    src_path = os.path.join(src_dir, file)
    dest_path = os.path.join(dest_dir, file)
    with Image.open(src_path) as img:
        img_resized = img.resize(
            (img.width // scale, img.height // scale), Image.BICUBIC
        )
        img_resized.save(dest_path)


def resize_and_save_images(files, src_dir, dest_dir, scale=4, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(resize_and_save_image, file, src_dir, dest_dir, scale)
            for file in files
        ]
        for future in futures:
            future.result()


def main():

    original_data_dir = "data/imgnet"
    data_dir = "data/imagenet"
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"
    test_dir = f"{data_dir}/test"

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    all_files = [f for f in os.listdir(original_data_dir) if f.endswith(".JPEG")]

    IMG_MIN_WIDTH = 400
    IMG_MAX_WIDTH = 500
    IMG_MIN_HEIGHT = 300
    IMG_MAX_HEIGHT = 400
    # run this loop only once
    for i, file_name in enumerate(tqdm(all_files, desc="Renaming files"), start=1):

        src_path = f"{original_data_dir}/{file_name}"
        with Image.open(src_path) as img:
            width, height = img.size
            if (width < IMG_MIN_WIDTH or width > IMG_MAX_WIDTH) or (
                height < IMG_MIN_HEIGHT or height > IMG_MAX_HEIGHT
            ):
                # delete the file
                os.remove(src_path)
                continue
        file_ext = file_name.split(".")[-1]
        new_file_name = f"{original_data_dir}/{i}.{file_ext}"
        os.rename(src_path, new_file_name)

    all_files = [f for f in os.listdir(original_data_dir) if f.endswith(".JPEG")]

    random.seed(42)
    random.shuffle(all_files)
    train_files, temp_files = train_test_split(
        all_files, test_size=(1 - train_ratio), random_state=42
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
    )

    max_workers = os.cpu_count() * 3

    scale_factors = [2, 4, 8]
    for scale in scale_factors:
        os.makedirs(f"{train_dir}/X{scale}/HR", exist_ok=True)
        os.makedirs(f"{train_dir}/X{scale}/LR", exist_ok=True)
        os.makedirs(f"{val_dir}/X{scale}/HR", exist_ok=True)
        os.makedirs(f"{val_dir}/X{scale}/LR", exist_ok=True)
        os.makedirs(f"{test_dir}/X{scale}/HR", exist_ok=True)
        os.makedirs(f"{test_dir}/X{scale}/LR", exist_ok=True)

        copy_files(
            train_files, original_data_dir, f"{train_dir}/X{scale}/HR", max_workers
        )
        copy_files(val_files, original_data_dir, f"{val_dir}/X{scale}/HR", max_workers)
        copy_files(
            test_files, original_data_dir, f"{test_dir}/X{scale}/HR", max_workers
        )

        train_source_dir = f"{original_data_dir}"
        train_dest_dir = f"{train_dir}/X{scale}/LR"
        resize_and_save_images(
            train_files, train_source_dir, train_dest_dir, scale, max_workers
        )

        val_source_dir = f"{original_data_dir}"
        val_dest_dir = f"{val_dir}/X{scale}/LR"
        resize_and_save_images(
            val_files, val_source_dir, val_dest_dir, scale, max_workers
        )

        test_source_dir = f"{original_data_dir}"
        test_dest_dir = f"{test_dir}/X{scale}/LR"
        resize_and_save_images(
            test_files, test_source_dir, test_dest_dir, scale, max_workers
        )


if __name__ == "__main__":
    main()
