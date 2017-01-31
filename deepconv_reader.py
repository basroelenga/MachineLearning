from __future__ import division

import scipy.misc as scim

import matplotlib.pyplot as plt

import numpy as np

import os


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        try:
            img = scim.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
        except IOError:
            pass
    return images, filenames

data = np.genfromtxt('results.dat', delimiter=" ")
scoresMain = np.genfromtxt('mainScores.dat', delimiter="    ", dtype=None)
scoresSpiral = np.genfromtxt('spiralScores.dat', delimiter="    ", dtype=None)
scoresEllipse = np.genfromtxt('ellipseScores.dat', delimiter="    ", dtype=None)

acc_eval_main = []
acc_eval_spiral = []
acc_eval_ellipse = []

for i in range(0, len(scoresMain)):
    
    acc = scoresMain[i][-1].split(' ')[2]
    acc_eval_main.append(acc)
    
    acc = scoresSpiral[i][-1].split(' ')[2]
    acc_eval_spiral.append(acc)
    
    acc = scoresEllipse[i][-1].split(' ')[2]
    acc_eval_ellipse.append(acc)

fig = plt.figure()
plt.plot(np.linspace(1, 400, 400), acc_eval_main)

plt.title("Ellipse/Spiral/Artifact accuracy rates")

plt.xlabel("Epoch")
plt.ylabel("Score")

fig = plt.figure()
plt.plot(np.linspace(1, 400, 400), acc_eval_spiral)

plt.title("Edge on/Not edge on accuracy rates")

plt.xlabel("Epoch")
plt.ylabel("Score")

fig = plt.figure()
plt.plot(np.linspace(1, 400, 400), acc_eval_ellipse)

plt.title("Round/Inbetween/Cigar shape accuracy rates")

plt.xlabel("Epoch")
plt.ylabel("Score")

plt.show()

#image_data, image_name = load_images_from_folder('/net/dataserver2/data/users/nobels/MachineLearning/galaxyzoo/1000imagesforbas')