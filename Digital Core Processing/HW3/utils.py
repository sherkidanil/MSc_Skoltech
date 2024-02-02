from matplotlib import pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

def plot_hist(img, hist_with_diff: bool = False):
    plt.hist(np.array(img).ravel(), bins=500)
    if hist_with_diff:
        hist, _ = np.histogram(img.ravel(), bins=range(0,257), density=False)
        diff = np.diff(hist) + hist[0]
        plt.plot(diff, label='1st derivation')
        plt.legend()
    plt.grid(True, color='#2A3459')
    plt.ylabel('frequency')
    plt.xlabel('intensity')
    plt.show()

def plot_img_and_hist(img, img_name='', hist_with_diff: bool = False):
    pylab.style.use('seaborn-dark')
    for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        plt.rcParams[param] = '#212946'
    for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
        plt.rcParams[param] = '0.9'

    pylab.figure(figsize=(15,5))
    pylab.subplot(121), pylab.imshow(img, cmap='gray'), pylab.title(img_name, size=18), pylab.grid(False)
    pylab.subplot(122), plot_hist(img, hist_with_diff), # pylab.yscale('log',basey=10)
    pylab.show()




