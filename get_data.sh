#!/bin/bash

# Configurations
# URL and file names for datasets
# Note: To use ImageNet, you need to provide the URL and zip file name from https://www.image-net.org
CELEB_URL="https://www.kaggle.com/api/v1/datasets/download/badasstechie/celebahq-resized-256x256"
CELEB_ZIP_FILE="celeb_a_hq.zip"

SET14_URL="https://www.kaggle.com/api/v1/datasets/download/guansuo/set14dataset"
SET14_ZIP_FILE="set14dataset.zip"

URBAN_URL="https://www.kaggle.com/api/v1/datasets/download/harshraone/urban100"
URBAN_ZIP_FILE="urban100.zip"

REALSR_URL="https://www.kaggle.com/api/v1/datasets/download/yashchoudhary/realsr-v3"
REALSR_ZIP_FILE="realsr-v3.zip"

DIV2k_URL="https://www.kaggle.com/api/v1/datasets/download/takihasan/div2k-dataset-for-super-resolution"
DIV2K_ZIP_FILE="div2k.zip"

IMGNET_URL="https://www.image-net.org/data"   # Provide your image-net url here
IMGNET_ZIP_FILE="ILSVRC2010_images_train.tar" # Provide your image-net zip file name here

# folder to save the datasets
DATA_DIR="data"

# Function to display usage of the script
usage() {
	echo "Usage: $0 [-i|--imagenet] [-c|--celeba] [-d|--div2k] [-r|--realsr] [-s|--set14] [-u|--urban100]"
	echo "  -i, --imagenet    Downloads the ImageNet dataset."
	echo "  -c, --celeba      Downloads the CelebA dataset."
	echo "  -d, --div2k       Downloads the DIV2K dataset."
	echo "  -r, --realsr      Downloads the RealSR dataset."
	echo "  -s, --set14       Downloads the Set14 dataset."
	echo "  -u, --urban100    Downloads the Urban100 dataset."
	exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
	-i | --imagenet)
		DOWNLOAD_IMGNET=true
		shift
		;;
	-c | --celeba)
		DOWNLOAD_CELEB=true
		shift
		;;
	-d | --div2k)
		DOWNLOAD_DIV2K=true
		shift
		;;
	-r | --realsr)
		DOWNLOAD_REALSR=true
		shift
		;;
	-s | --set14)
		DOWNLOAD_SET14=true
		shift
		;;
	-u | --urban100)
		DOWNLOAD_URBAN=true
		shift
		;;
	*)
		usage
		;;
	esac
done

# Check if at least one option is provided
if [ -z "$DOWNLOAD_CELEB" ] && [ -z "$DOWNLOAD_SET14" ] && [ -z "$DOWNLOAD_URBAN" ] &&
	[ -z "$DOWNLOAD_REALSR" ] && [ -z "$DOWNLOAD_DIV2K" ] && [ -z "$DOWNLOAD_IMGNET" ]; then
	usage
fi

# Check if the data directory exists, if not create it
if [ ! -d $DATA_DIR ]; then
	echo "Creating data directory..."
	mkdir -p $DATA_DIR
else
	echo "Data directory already exists. Skipping creation."
fi

# installing curl and unzip if not installed
if ! command -v curl &>/dev/null; then
	echo "curl could not be found, installing..."
	sudo apt-get update
	sudo apt-get install -y curl
fi
if ! command -v unzip &>/dev/null; then
	echo "unzip could not be found, installing..."
	sudo apt-get update
	sudo apt-get install -y unzip
fi

# CELEB-A-HQ
download_celeb() {
	echo "Downloading Celeb-A-HQ dataset..."
	if [ ! -f "$DATA_DIR/$CELEB_ZIP_FILE" ]; then
		echo "Downloading the zip file..."
		curl -L -o "$DATA_DIR/$CELEB_ZIP_FILE" "$CELEB_URL"
		if [ $? -ne 0 ]; then
			echo "Error: Failed to download Celeb-A-HQ dataset."
			return 1 # Exit the function with an error code
		fi
	else
		echo "Celeb-A-HQ zip file already exists. Skipping download."
	fi

	echo "Unzipping the dataset..."
	unzip -o "$DATA_DIR/$CELEB_ZIP_FILE" "celeba_hq_256/*" -d "$DATA_DIR" >/dev/null 2>&1
	if [ $? -ne 0 ]; then
		echo "Error: Failed to unzip Celeb-A-HQ dataset."
		return 1
	fi

	if [ -d "$DATA_DIR/celeb" ]; then
		echo "Celeb-A-HQ directory already exists. Removing old directory..."
		rm -rf "$DATA_DIR/celeb"
	fi

	echo "Splitting data into train, test, and validation sets..."
	python scripts/data_scripts/divide_data_celeb.py
	if [ $? -ne 0 ]; then
		echo "Error: Failed to split Celeb-A-HQ dataset."
		return 1
	fi

	echo "Cleaning up Celeb files..."
	rm -rf "$DATA_DIR/$CELEB_ZIP_FILE"
	if [ $? -ne 0 ]; then
		echo "Error: Failed to clean up Celeb-A-HQ zip file."
		return 1
	fi
	rm -rf "$DATA_DIR/celeba_hq_256"
	if [ $? -ne 0 ]; then
		echo "Error: Failed to clean up Celeb-A-HQ unzipped files."
		return 1
	fi

	echo "Celeb-A-HQ dataset downloaded and processed successfully."
	return 0 # Indicate success
}

# SET14
download_set14() {
	echo "Downloading Set14 dataset..."
	if [ ! -f "$DATA_DIR/$SET14_ZIP_FILE" ]; then
		echo "Downloading the zip file..."
		curl -L -o "$DATA_DIR/$SET14_ZIP_FILE" "$SET14_URL"
		if [ $? -ne 0 ]; then
			echo "Error: Failed to download Set14 dataset."
			return 1
		fi
	else
		echo "Set14 zip file already exists. Skipping download."
	fi

	echo "Unzipping the dataset..."
	mkdir -p "$DATA_DIR/set14"
	unzip -o "$DATA_DIR/$SET14_ZIP_FILE" "Set14/*" -d "$DATA_DIR" >/dev/null 2>&1
	if [ $? -ne 0 ]; then
		echo "Error: Failed to unzip Set14 dataset."
		return 1
	fi

	echo "Organizing dataset into directories..."
	declare -A NEW_DIRS=(
		["image_SRF_2"]="X2"
		["image_SRF_3"]="X3"
		["image_SRF_4"]="X4"
	)

	for key in "${!NEW_DIRS[@]}"; do
		mkdir -p "$DATA_DIR/Set14/${NEW_DIRS[$key]}/HR"
		mkdir -p "$DATA_DIR/Set14/${NEW_DIRS[$key]}/LR"
	done

	# Copy HR and LR files to their respective directories
	for key in "${!NEW_DIRS[@]}"; do
		for file in "$DATA_DIR/Set14/$key"/*; do
			if [[ $file == *_HR.png ]]; then
				dest_dir="$DATA_DIR/Set14/${NEW_DIRS[$key]}/HR"
				file_count=$(ls -1q "$dest_dir" | wc -l)
				new_file_name="$dest_dir/$(($file_count + 1)).png"
				mv "$file" "$new_file_name"
			elif [[ $file == *_LR.png ]]; then
				dest_dir="$DATA_DIR/Set14/${NEW_DIRS[$key]}/LR"
				file_count=$(ls -1q "$dest_dir" | wc -l)
				new_file_name="$dest_dir/$(($file_count + 1)).png"
				mv "$file" "$new_file_name"
			fi
		done
	done

	for key in "${!NEW_DIRS[@]}"; do
		rmdir "$DATA_DIR/Set14/$key"
	done

    echo "Cleaning up Set14 files..."
    rm -rf "$DATA_DIR/$SET14_ZIP_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clean up Set14 zip file."
        return 1
    fi

	echo "Set14 dataset downloaded and processed successfully."
	return 0
}

# URBAN100
download_urban() {
	echo "Downloading Urban100 dataset..."
	if [ ! -f "$DATA_DIR/$URBAN_ZIP_FILE" ]; then
		echo "Downloading the zip file..."
		curl -L -o "$DATA_DIR/$URBAN_ZIP_FILE" "$URBAN_URL"
		if [ $? -ne 0 ]; then
			echo "Error: Failed to download Urban100 dataset."
			return 1
		fi
	else
		echo "Urban100 zip file already exists. Skipping download."
	fi

	echo "Unzipping the dataset..."
	unzip -o "$DATA_DIR/$URBAN_ZIP_FILE" "Urban 100/*" -d "$DATA_DIR" >/dev/null 2>&1
	if [ $? -ne 0 ]; then
		echo "Error: Failed to unzip Urban100 dataset."
		return 1
	fi

	echo "Creating directories for Urban100..."
	mkdir -p "$DATA_DIR/urban100" "$DATA_DIR/urban100/X2/HR" "$DATA_DIR/urban100/X2/LR" "$DATA_DIR/urban100/X4/HR" "$DATA_DIR/urban100/X4/LR"

	echo "Organizing files into directories..."
	mv "$DATA_DIR/Urban 100/X2 Urban100/X2/HIGH X2 Urban"/* "$DATA_DIR/urban100/X2/HR"
	mv "$DATA_DIR/Urban 100/X2 Urban100/X2/LOW X2 Urban"/* "$DATA_DIR/urban100/X2/LR"
	mv "$DATA_DIR/Urban 100/X4 Urban100/X4/HIGH x4 URban100"/* "$DATA_DIR/urban100/X4/HR"
	mv "$DATA_DIR/Urban 100/X4 Urban100/X4/LOW x4 URban100"/* "$DATA_DIR/urban100/X4/LR"
	rm -rf "$DATA_DIR/Urban 100"

	echo "Renaming files..."
	dirs=(
		"$DATA_DIR/urban100/X2/HR"
		"$DATA_DIR/urban100/X2/LR"
		"$DATA_DIR/urban100/X4/HR"
		"$DATA_DIR/urban100/X4/LR"
	)

	for file_dir in "${dirs[@]}"; do
		for file in "$file_dir"/*; do
			new_name=$(echo "$file" | sed -E 's/img_([0-9]+)_.*/\1.png/')
			mv "$file" "$new_name"
		done
	done

    echo "Cleaning up Urban100 files..."
    rm -rf "$DATA_DIR/$URBAN_ZIP_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clean up Urban100 zip file."
        return 1
    fi

	echo "Urban100 dataset downloaded and processed successfully."
	return 0
}

# REALSR
download_realsr() {
	echo "Downloading RealSR dataset..."
	if [ ! -f "$DATA_DIR/$REALSR_ZIP_FILE" ]; then
		echo "Downloading the zip file..."
		curl -L -o "$DATA_DIR/$REALSR_ZIP_FILE" "$REALSR_URL"
		if [ $? -ne 0 ]; then
			echo "Error: Failed to download RealSR dataset."
			return 1
		fi
	else
		echo "RealSR zip file already exists. Skipping download."
	fi

	echo "Unzipping the dataset..."
	unzip -o "$DATA_DIR/$REALSR_ZIP_FILE" "RealSR(V3)/*" -d "$DATA_DIR" >/dev/null 2>&1
	if [ $? -ne 0 ]; then
		echo "Error: Failed to unzip RealSR dataset."
		return 1
	fi

	echo "Creating directories for RealSR..."
	mkdir -p "$DATA_DIR/realsr_canon/train" "$DATA_DIR/realsr_canon/test"
	mkdir -p "$DATA_DIR/realsr_nikon/train" "$DATA_DIR/realsr_nikon/test"

	declare -A NEW_DIRS=(
		["Canon/Test/2"]="Canon/Test/X2"
		["Canon/Train/2"]="Canon/Train/X2"
		["Canon/Test/3"]="Canon/Test/X3"
		["Canon/Train/3"]="Canon/Train/X3"
		["Canon/Test/4"]="Canon/Test/X4"
		["Canon/Train/4"]="Canon/Train/X4"
		["Nikon/Test/2"]="Nikon/Test/X2"
		["Nikon/Train/2"]="Nikon/Train/X2"
		["Nikon/Test/3"]="Nikon/Test/X3"
		["Nikon/Train/3"]="Nikon/Train/X3"
		["Nikon/Test/4"]="Nikon/Test/X4"
		["Nikon/Train/4"]="Nikon/Train/X4"
	)

	echo "Organizing files into directories for canon..."
	for key in "${!NEW_DIRS[@]}"; do
		mkdir -p "$DATA_DIR/RealSR(V3)/${NEW_DIRS[$key]}/HR" "$DATA_DIR/RealSR(V3)/${NEW_DIRS[$key]}/LR"
		for file in "$DATA_DIR/RealSR(V3)/$key"/*; do
			if [[ $file == *_HR*.png ]]; then
				dest_dir="$DATA_DIR/RealSR(V3)/${NEW_DIRS[$key]}/HR"
				file_count=$(ls -1q "$dest_dir" | wc -l)
				new_file_name="$dest_dir/$(($file_count + 1)).png"
				mv "$file" "$new_file_name"
			elif [[ $file == *_LR*.png ]]; then
				dest_dir="$DATA_DIR/RealSR(V3)/${NEW_DIRS[$key]}/LR"
				file_count=$(ls -1q "$dest_dir" | wc -l)
				new_file_name="$dest_dir/$(($file_count + 1)).png"
				mv "$file" "$new_file_name"
			fi
		done
		rm -rf "$DATA_DIR/RealSR(V3)/$key"
	done

	echo "Moving processed files to final directories..."
	mv "$DATA_DIR/RealSR(V3)/Canon/Train"/* "$DATA_DIR/realsr_canon/train"
	mv "$DATA_DIR/RealSR(V3)/Canon/Test"/* "$DATA_DIR/realsr_canon/test"
	mv "$DATA_DIR/RealSR(V3)/Nikon/Train"/* "$DATA_DIR/realsr_nikon/train"
	mv "$DATA_DIR/RealSR(V3)/Nikon/Test"/* "$DATA_DIR/realsr_nikon/test"

	echo "Creating additional directories for RealSR..."
	mkdir -p "$DATA_DIR/realsr_canon/X2/HR" "$DATA_DIR/realsr_canon/X2/LR" "$DATA_DIR/realsr_canon/X3/HR" "$DATA_DIR/realsr_canon/X3/LR" "$DATA_DIR/realsr_canon/X4/HR" "$DATA_DIR/realsr_canon/X4/LR"
	mkdir -p "$DATA_DIR/realsr_nikon/X2/HR" "$DATA_DIR/realsr_nikon/X2/LR" "$DATA_DIR/realsr_nikon/X3/HR" "$DATA_DIR/realsr_nikon/X3/LR" "$DATA_DIR/realsr_nikon/X4/HR" "$DATA_DIR/realsr_nikon/X4/LR"

	declare -A NEW_DIRS_FINAL=(
		["Canon/Test/2"]="realsr_canon/X2"
		["Canon/Train/2"]="realsr_canon/X2"
		["Canon/Test/3"]="realsr_canon/X3"
		["Canon/Train/3"]="realsr_canon/X3"
		["Canon/Test/4"]="realsr_canon/X4"
		["Canon/Train/4"]="realsr_canon/X4"
		["Nikon/Test/2"]="realsr_nikon/X2"
		["Nikon/Train/2"]="realsr_nikon/X2"
		["Nikon/Test/3"]="realsr_nikon/X3"
		["Nikon/Train/3"]="realsr_nikon/X3"
		["Nikon/Test/4"]="realsr_nikon/X4"
		["Nikon/Train/4"]="realsr_nikon/X4"
	)

	echo "Organizing files into final directories for nikon..."
	for key in "${!NEW_DIRS_FINAL[@]}"; do
		for file in "$DATA_DIR/RealSR(V3)/$key"/*; do
			if [[ $file == *_HR*.png ]]; then
				dest_dir="$DATA_DIR/${NEW_DIRS_FINAL[$key]}/HR"
				file_count=$(ls -1q "$dest_dir" | wc -l)
				new_file_name="$dest_dir/$(($file_count + 1)).png"
				mv "$file" "$new_file_name"
			elif [[ $file == *_LR*.png ]]; then
				dest_dir="$DATA_DIR/${NEW_DIRS_FINAL[$key]}/LR"
				file_count=$(ls -1q "$dest_dir" | wc -l)
				new_file_name="$dest_dir/$(($file_count + 1)).png"
				mv "$file" "$new_file_name"
			fi
		done
	done

    echo "Cleaning up RealSR files..."
	rm -rf "$DATA_DIR/REALSR(V3)"
	rm -rf "$DATA_DIR/$REALSR_ZIP_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clean up RealSR zip file."
        return 1
    fi

	echo "RealSR dataset downloaded and processed successfully."
	return 0
}

# DIV2K
download_div2k() {
	echo "Downloading DIV2K dataset..."
	if [ ! -f "$DATA_DIR/$DIV2K_ZIP_FILE" ]; then
		echo "Downloading the zip file..."
		curl -L -o "$DATA_DIR/$DIV2K_ZIP_FILE" "$DIV2k_URL"
		if [ $? -ne 0 ]; then
			echo "Error: Failed to download DIV2K dataset."
			return 1
		fi
	else
		echo "DIV2K zip file already exists. Skipping download."
	fi

	echo "Unzipping the dataset..."
	unzip -o "$DATA_DIR/$DIV2K_ZIP_FILE" "Dataset/*" -d "$DATA_DIR" >/dev/null 2>&1
	if [ $? -ne 0 ]; then
		echo "Error: Failed to unzip DIV2K dataset."
		return 1
	fi

	echo "Creating directories for DIV2K..."
	mkdir -p "$DATA_DIR/div2k/train" "$DATA_DIR/div2k/test"
	mkdir -p "$DATA_DIR/Dataset/train/X2/HR" "$DATA_DIR/Dataset/train/X2/LR" "$DATA_DIR/Dataset/train/X4/HR" "$DATA_DIR/Dataset/train/X4/LR"
	mkdir -p "$DATA_DIR/Dataset/test/X2/HR" "$DATA_DIR/Dataset/test/X2/LR" "$DATA_DIR/Dataset/test/X4/HR" "$DATA_DIR/Dataset/test/X4/LR"

	declare -A NEW_DIRS=(
		["DIV2K_train_LR_bicubic/X2"]="train/X2/LR"
		["DIV2K_train_LR_bicubic_X4/X4"]="train/X4/LR"
		["DIV2K_valid_LR_bicubic/X2"]="test/X2/LR"
		["DIV2K_valid_LR_bicubic_X4/X4"]="test/X4/LR"
	)

	echo "Organizing LR files..."
	for key in "${!NEW_DIRS[@]}"; do
		for file in "$DATA_DIR/Dataset/$key"/*; do
			dest_dir="$DATA_DIR/Dataset/${NEW_DIRS[$key]}"
			file_name=$(basename "$file")
			new_file_name="${file_name%??.png}.png"
			mv "$file" "$dest_dir/$new_file_name"
		done
	done

	echo "Copying HR files..."
	for file in "$DATA_DIR/Dataset/DIV2K_train_HR"/*; do
		dest_dir="$DATA_DIR/Dataset/train/X2/HR"
		file_name=$(basename "$file")
		cp "$file" "$dest_dir/$file_name"
		dest_dir="$DATA_DIR/Dataset/train/X4/HR"
		cp "$file" "$dest_dir/$file_name"
	done

	for file in "$DATA_DIR/Dataset/DIV2K_valid_HR"/*; do
		dest_dir="$DATA_DIR/Dataset/test/X2/HR"
		file_name=$(basename "$file")
		cp "$file" "$dest_dir/$file_name"
		dest_dir="$DATA_DIR/Dataset/test/X4/HR"
		cp "$file" "$dest_dir/$file_name"
	done

	echo "Moving processed files to final directories..."
	mv "$DATA_DIR/Dataset/train"/* "$DATA_DIR/div2k/train"
	mv "$DATA_DIR/Dataset/test"/* "$DATA_DIR/div2k/test"
	rm -rf "$DATA_DIR/Dataset"

    echo "Cleaning up DIV2K files..."
    rm -rf "$DATA_DIR/$DIV2K_ZIP_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clean up DIV2K zip file."
        return 1
    fi

	echo "DIV2K dataset downloaded and processed successfully."
	return 0
}

# image-net
download_imgnet() {
	echo "Downloading ImageNet dataset..."
	if [ ! -f "$DATA_DIR/$IMGNET_ZIP_FILE" ]; then
		echo "Downloading the zip file..."
		wget "$IMGNET_URL" -O "$DATA_DIR/$IMGNET_ZIP_FILE"
		if [ $? -ne 0 ]; then
			echo "Error: Failed to download ImageNet dataset."
			return 1
		fi
	else
		echo "ImageNet zip file already exists. Skipping download."
	fi

	echo "Extracting the main tar file..."
	mkdir -p "$DATA_DIR/imgnet"
	tar -xvf "$DATA_DIR/$IMGNET_ZIP_FILE" -C "$DATA_DIR/imgnet" >/dev/null 2>&1
	if [ $? -ne 0 ]; then
		echo "Error: Failed to extract ImageNet tar file."
		return 1
	fi

	echo "Extracting nested tar files..."
	for tar_file in "$DATA_DIR/imgnet"/*.tar; do
		tar -xf "$tar_file" -C "$DATA_DIR/imgnet" >/dev/null 2>&1
		if [ $? -ne 0 ]; then
			echo "Error: Failed to extract nested tar file: $tar_file"
			return 1
		fi
	done

	echo "Splitting data into train, test, and validation sets..."
	python scripts/data_scripts/divide_data_imgnet.py
	if [ $? -ne 0 ]; then
		echo "Error: Failed to split ImageNet dataset."
		return 1
	fi

	echo "Cleaning up ImageNet files..."
	rm -rf "$DATA_DIR/imgnet"
	rm -rf "$DATA_DIR/$tar_file"

	echo "ImageNet dataset downloaded and processed successfully."
	return 0
}

# Execute downloads based on arguments
echo "Starting downloads and processing..."

[ "$DOWNLOAD_CELEB" = true ] && download_celeb
[ "$DOWNLOAD_SET14" = true ] && download_set14
[ "$DOWNLOAD_URBAN" = true ] && download_urban
[ "$DOWNLOAD_REALSR" = true ] && download_realsr
[ "$DOWNLOAD_DIV2K" = true ] && download_div2k
[ "$DOWNLOAD_IMGNET" = true ] && download_imgnet

echo "Finished"
