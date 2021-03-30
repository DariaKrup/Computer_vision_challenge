import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class Transform:
    def __init__(self, obj):
        self.scale = obj['scale'] if 'scale' in obj else 1
        self.angle = obj['angle'] * np.pi / 180.0 if 'angle' in obj else 0
        self.dx = obj['dx'] if 'dx' in obj else 0
        self.dy = obj['dy'] if 'dy' in obj else 0


def draw(inpt, shape, transform: Transform, color=(125, 126, 125)):
    new_shape = shape.copy().astype(np.float)
    # Scale
    new_shape *= transform.scale

    # Rotation
    tmp = new_shape.copy()
    for i in (0, 1):
        new_shape[:, i] = np.cos(transform.angle) * tmp[:, i] \
                          - ((-1) ** i) * np.sin(transform.angle) * tmp[:, 1 - i]

    # Shift
    new_shape[:, 0] += transform.dx
    new_shape[:, 1] += transform.dy

    # cv.fillPoly(gt, [new_shape.astype(np.int32)], color)

    cv.polylines(inpt, [new_shape.astype(np.int32)], True, color)


def plot_results(found_figures, basis_figures, image_src):
    image_src_copy = image_src.copy()

    for figure in found_figures:
        basis_figure_id = figure[0]
        scale = figure[3]
        rotation_angle = figure[4]
        shift_x = figure[1]
        shift_y = figure[2]
        basis_figure = basis_figures[basis_figure_id]

        obj = dict(scale=scale, angle=rotation_angle, dx=shift_x, dy=shift_y)
        transform = Transform(obj)

        draw(image_src_copy,
             np.array(basis_figure, dtype=np.int32),
             transform)

    f, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].set_title("src result")
    ax[1].set_title("src original")
    ax[0].imshow(image_src_copy, cmap='gray')
    ax[1].imshow(image_src, cmap='gray')


"""class Transform:
    def __init__(self, obj):
        self.polygon = obj['polygon'] if 'polygon' in obj else []
        self.scale = obj['scale'] if 'scale' in obj else 1
        self.angle = obj['angle'] * np.pi / 180.0 if 'angle' in obj else 0
        self.dx = obj['dx'] if 'dx' in obj else 0
        self.dy = obj['dy'] if 'dy' in obj else 0

    def draw(self, inpt, gt, color=255):
        assert (inpt.shape == gt.shape)

        new_shape = self.polygon.copy().astype(np.float)
        # Scale
        new_shape *= self.scale

        # Rotation
        tmp = new_shape.copy()
        for i in [0, 1]:
            new_shape[:, i] = np.cos(self.angle) * tmp[:, i] \
                             - ((-1)** i) * np.sin(self.angle) * tmp[:, 1 - i]

        #Shift
        new_shape[:, 0] += self.dx
        new_shape[:, 1] += self.dy

        cv2.fillPoly(gt, [new_shape.astype(np.int32)], color)
        cv2.polylines(inpt, [new_shape.astype(np.int32)], True, color)

def draw_all(shape, figures, img_inpt, img_gt):
    for figure in figures:
        basis_figure_id = figure[0]
        scale = figure[1]
        rotation_angle = figure[2]
        shift_x = figure[3]
        shift_y = figure[4]
        fig = Transform({'polygon': shape[basis_figure_id], 'scale': scale, 'angle': rotation_angle, 'dx': shift_x, 'dy': shift_y})
        fig.draw(img_inpt, img_gt)
    cv2.imshow("test_img_inpt", img_inpt)
    cv2.waitKey(0)
    cv2.imshow("test_img_gt", img_gt)
    cv2.waitKey(0)
    f, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].set_title("src result")
    ax[1].set_title("src original")
    ax[0].imshow(img_inpt, cmap='gray')
    ax[1].imshow(img_gt, cmap='gray')
    plt.show()"""

