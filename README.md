# Galaxies on an adventure in the magical world of machine learning

## File names and their function

* augmentation.py: program to do the augmentation of the images, this file consists of all required functions to do the augmentation. (the augmentation.pyc is a compiled version of this program).
* bashscript.sh: Script that runs 9 processes instantaneously on all available cores, for the main program (max 9 cores).
* main.py: Main program for the data augmentation, that runs all augmentation instanteously.
* make_dataset.py: Program that makes a dataset for all available data to train the deeplearning network.
* test.txt: Dataset of the test names of the test objects.
* training.txt: Dataset of the training data.

## Some useful github commands:

- Always when you start coding, pull the existing code from the repository so everything is in sync with eachother
command: git pull https://github.com/basroelenga/MachineLearning.git dev

- If you do not want to type "https://github.com/basroelenga/MachineLearning.git" all the time when pushing/pulling
command: git remote add https://github.com/basroelenga/MachineLearning.git

- Now a push command looks like this
command: git pull origin dev

where origin --> https://github.com/basroelenga/MachineLearning.git

- When you have created a new file make sure to add it to the git tracking list by using
command: git add filename

or if you want to add all files that are not yet added (add all untracked files)
command: git add -A

- When you are done with some work in a file you have to commit the changes by using
command: git commit -m "Some messages here, is mandatory" filename

or just commit everything at once
command: git commit -am "mandatory messsage"

- Then when you are done, push all the changes to the repository by using
command: git push origin dev

If this gives some merging errors, it means that somebody added the repository since the time you pulled from it
Just pull the repository again (second command on the list) and then push again
