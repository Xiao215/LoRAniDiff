"""
This script provides an interface for downloading and optionally resizing specific datasets.
Currently supported datasets include 'pixiv' and 'textcaps'. The user can choose which dataset
to download and, for image datasets, whether to resize the images.
"""

import subprocess
import os


def main():
    """Main function to handle user inputs for dataset download and processing."""
    choice = (
        input("Which dataset do you want to download? (pixiv [p]/textcaps [t]): ")
        .strip()
        .lower()
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if choice in ("pixiv", "p"):
        dataset_dir = os.path.join(base_dir, "data/pixiv")
        if os.path.exists(dataset_dir):
            print(
                f"The folder {dataset_dir} already exists. Please remove it before downloading the dataset."
            )
            return
        resize = (
            input(
                "Do you want to resize the images? If no, default is 360x360. (yes [y]/NO [n]): "
            )
            .strip()
            .lower()
        )
        pixiv_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "pixiv", "pixiv_dataset.py"
        )

        if resize in ("yes", "y"):
            size = input(
                "Enter the size to which you want the images resized (e.g., 512): "
            ).strip()
            subprocess.run(["python3", pixiv_script_path, "--resize", size], check=True)
        elif resize in ("no", "n", ""):
            subprocess.run(["python3", pixiv_script_path], check=True)
        else:
            print("Invalid choice. Please enter 'yes' or 'no'.")
            return

    elif choice in ("textcaps", "t"):
        dataset_dir = os.path.join(base_dir, "data", "textcaps")
        if os.path.exists(dataset_dir):
            print(
                f"The folder {dataset_dir} already exists. Please remove it before downloading the dataset."
            )
            return
        textcaps_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "textcaps", "textcaps.sh"
        )
        subprocess.run([textcaps_script_path], shell=True, check=True)
    else:
        print("Invalid choice. Please enter 'pixiv (p)' or 'textcaps (t)'.")
        return


if __name__ == "__main__":
    main()
