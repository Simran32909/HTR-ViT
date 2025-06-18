#!/bin/bash

# Prepare and organize data for the Rimes dataset
folder_name=$1
dataset="rimes"

echo "Preparing $dataset dataset"
mkdir -p $folder_name/htr_datasets
mkdir -p $folder_name/$dataset

# Decompress the dataset
unzip -q $folder_name/RIMES-2011-Lines.zip -d $folder_name/$dataset 

# Rename the split files to the expected train.txt, val.txt, and test.txt
mv $folder_name/$dataset/RIMES-2011-Lines/Sets/TrainLines.txt $folder_name/$dataset/RIMES-2011-Lines/Sets/train.txt
mv $folder_name/$dataset/RIMES-2011-Lines/Sets/ValidationLines.txt $folder_name/$dataset/RIMES-2011-Lines/Sets/val.txt
mv $folder_name/$dataset/RIMES-2011-Lines/Sets/TestLines.txt $folder_name/$dataset/RIMES-2011-Lines/Sets/test.txt

# Remove any empty __MACOSX directories if they exist
find $folder_name/$dataset/ -name "__MACOSX" -type d -exec rm -rf {} \;

# Move the processed dataset to the final htr_datasets directory
mv $folder_name/$dataset/ $folder_name/htr_datasets/
