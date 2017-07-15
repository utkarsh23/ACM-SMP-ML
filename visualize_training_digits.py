import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from random import randint 
from matplotlib.widgets import Button

class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        if self.ind == len(train_images):
            self.ind = 0
        existing_axis.set_title("Training Image " + str(self.ind + 1))
        existing_axis.imshow(train_images[self.ind], cmap='gray')
        fig.canvas.draw()

    def prev(self, event):
        self.ind -= 1
        if self.ind == -1:
            self.ind = len(train_images) - 1
        existing_axis.set_title("Training Image " + str(self.ind + 1))
        existing_axis.imshow(train_images[self.ind], cmap='gray')
        fig.canvas.draw()

train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')

fig = plt.figure()
fig.canvas.set_window_title('Training Dataset of Handwritten Digits')
existing_axis = fig.gca()
existing_axis.imshow(train_images[0], cmap='gray')
existing_axis.set_title("Training Image 1")

callback = Index()
axprev = plt.axes([0.85, 0.45, 0.1, 0.05])
axnext = plt.axes([0.05, 0.45, 0.1, 0.05])
bnext = Button(axprev, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axnext, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()