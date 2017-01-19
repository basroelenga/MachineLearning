#!/usr/bin/env bash

module load anaconda/4.2.0
echo 'welcome to the demo'
echo ' '
echo 'The first step  is the  creation  of the training data'
echo 'Given  that  we  have  already  a set of galaxies, but '
echo 'this by far is not enough, we need to augment the data'
echo 'this is done in this part, '
read -rsp $'Press any key to continue...\n' -n1 key
echo 'Starting the image augmentation program'
rm ./galaxyzoo/images_augmentation/*.jpg
./ML_augmentation_001.py
echo 'Finished the image augmentation'
echo ' '
echo ' '
echo 'The second step is attaching a classification to all images'
echo 'This  was  done  with  a  preknown  classification  of  all '
echo 'individual  galaxies,  and  renaming  the  names  we   only '
echo 'made a difference between objects  which are  simply smooth'
echo 'or rounded, galaxy  with features and artifacts,  artifacts'
echo 'are stars and technical errors'
echo 'We can run the algorithm right now, '
read -rsp $'Press any key to continue...\n' -n1 key
echo 'Starting renaming the images'
./easyloadtraining.py
echo 'Finished renaming the training data'
echo ' '
echo ' '
echo 'in the next  part we will  make a  dataset of the  images'
echo 'This file will be immediately fed to the machine learning '
read -rsp $'Press any key to continue...\n' -n1 key
echo 'Starting creating the dataset and machine learning x 3 '
echo 'Finished creating the dataset and doing the machine learning'
python classification_c.py 100
python classification_c.py 100
python classification_c.py 100
