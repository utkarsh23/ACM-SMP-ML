import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from random import randint 
from matplotlib.widgets import Button

class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        if self.ind == len(t10k_images):
            self.ind = 0
        existing_axis.set_title("Testing Image " + str(self.ind + 1))
        existing_axis.imshow(t10k_images[self.ind], cmap='gray')
        fig.canvas.draw()

    def prev(self, event):
        self.ind -= 1
        if self.ind == -1:
            self.ind = len(t10k_images) - 1
        existing_axis.set_title("Testing Image " + str(self.ind + 1))
        existing_axis.imshow(t10k_images[self.ind], cmap='gray')
        fig.canvas.draw()

t10k_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')

fig = plt.figure()
fig.canvas.set_window_title('Testing Dataset of Handwritten Digits')
existing_axis = fig.gca()
existing_axis.imshow(t10k_images[0], cmap='gray')
existing_axis.set_title("Testing Image 1")

callback = Index()
axprev = plt.axes([0.83, 0.45, 0.12, 0.05])
axnext = plt.axes([0.03, 0.45, 0.12, 0.05])
bnext = Button(axprev, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axnext, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()