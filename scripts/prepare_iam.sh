#!/bin/bash

# Prepare and organize data for training, validation and testing for the IAM dataset
# It will create (or modify) a split file for each dataset and each stage (train, val, test)
folder_name=$1 # It is the original folder name in which original files are stored
datasets="iam"

mkdir -p $folder_name/htr_datasets # Real dataset folder
mkdir -p $folder_name/synth # Synth dataset folder

# Organize real HTR datasets
for dataset in $datasets
do
  echo "Preparing $dataset dataset"
  mkdir -p $folder_name/$dataset

  # IAM
  if [ $dataset == "iam" ]
  then
    # Descompress the dataset (lines.tar, xml.tar and splits.zip) 
    mkdir -p $folder_name/$dataset/iam/
    mkdir -p $folder_name/$dataset/iam/lines
    mkdir -p $folder_name/$dataset/iam/xml
    mkdir -p $folder_name/$dataset/iam/splits
    # Mantain the folder structure
    tar -xf $folder_name/lines.tgz -C $folder_name/$dataset/iam/lines
    tar -xf $folder_name/xml.tgz -C $folder_name/$dataset/iam/xml
    unzip -q $folder_name/splits.zip -d $folder_name/$dataset/iam/

    # Create a train.txt, val.txt and test.txt file and paste the content of train.uttlist, validation.uttlist and test.uttlist
    mv $folder_name/$dataset/iam/splits/train.uttlist $folder_name/$dataset/iam/splits/train.txt
    mv $folder_name/$dataset/iam/splits/validation.uttlist $folder_name/$dataset/iam/splits/val.txt
    mv $folder_name/$dataset/iam/splits/test.uttlist $folder_name/$dataset/iam/splits/test.txt
  fi
  
  # Remove all paths that contain __MACOSX in $folder_name/$datasettaset
  find $folder_name/$dataset/ -name "__MACOSX" -type d -exec rm -rf {} \;

  # Move all to the real dataset folder
  mv $folder_name/$dataset/ $folder_name/htr_datasets

done

# Organize synthetic HTR datasets
# unzip $folder_name/synth-data.zip -d $folder_name/

# Remove synth-data folder to clean the structure
# rm -rf $folder_name/synth-data

# Create a folder for the fonts
# mkdir -p $folder_name/synth/fonts 