import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


def view3d_image(input_im, cmap='gray', title='', spacing=None, slice_axis=0, filename=""):
    if spacing is None:
        spacing = [1, 1, 1]
    if isinstance(input_im, sitk.Image):
        input_numpy = sitk.GetArrayFromImage(input_im)
        spacing = input_im.GetSpacing()[::-1]
    else:
        input_numpy = input_im
    if slice_axis == 0:
        aspect = spacing[1] / spacing[2]
        # needs to be checked
    elif slice_axis == 1:
        input_numpy = np.transpose(input_numpy, [1, 0, 2])
        aspect = spacing[0] / spacing[2]
        # needs to be checked
    elif slice_axis == 2:
        input_numpy = np.transpose(input_numpy, [2, 0, 1])
        aspect = spacing[0] / spacing[1]
        # needs to be checked
    else:
        raise ValueError('slice_axis = '+str(slice_axis)+', but it should be in range of [0, 1, 2]')
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, input_numpy, cmap=cmap, title=title, aspect=aspect)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

    if filename != "":
        fig.savefig(filename)
        plt.close('all')

    else:
        plt.show()

    


class IndexTracker(object):
    # https://matplotlib.org/2.1.2/gallery/animation/image_slices_viewer.html
    def __init__(self, ax, X, cmap='gray', title='', aspect=1):
        self.ax = ax
        ax.set_title(title)

        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[self.ind, :, :], cmap=cmap, aspect=aspect)
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()




# # mhd_file = './data/CREATIS/0/10_R_enhanced.mha'
# mhd_file = './data/DIRLAB/mha/case10/case10_T00_R.mha'
# image = sitk.ReadImage(mhd_file)
# image_array = sitk.GetArrayFromImage(image)
# width, height, depth = image_array.shape

# # You can also access the spacing and origin information from the image
# spacing = image.GetSpacing()
# origin = image.GetOrigin()


# view3d_image(image_array, slice_axis=0)
# import ipdb; ipdb.set_trace()