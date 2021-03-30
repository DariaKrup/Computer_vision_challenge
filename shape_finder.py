import argparse
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_opening
from skimage.filters import threshold_otsu
from scipy.spatial import ConvexHull
from calc_geometry import *
from image_check import *


def find_shape_points_list(src):
    img = src.copy()
    shape_points_list = []

    tmpList = []
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] != 0:
                points = [(x, y)]
                tmpList.append((x, y))
                while len(tmpList):
                    elem = tmpList.pop()
                    for u in range(max(elem[0] - 1, 0), min(elem[0] + 2, img.shape[0] + 1)):
                        for v in range(max(elem[1] - 1, 0), min(elem[1] + 2, img.shape[1] + 1)):
                            if img[u, v] != 0:
                                img[u, v] = 0
                                if (u, v) not in tmpList:
                                    points.append((u, v))
                                    tmpList.append((u, v))
                shape_points_list.append(points)

    return shape_points_list


def delete_noise(img):
    dst = img.copy()
    threshold = threshold_otsu(dst)
    dst = dst > threshold

    dst = binary_fill_holes(dst)
    dst = binary_opening(dst)

    return dst


def find_angle_and_shift(polygon, shape, scale):
    opt_alpha = 0
    opt_shift = (0, 0)
    min_err = np.Inf
    for alpha in range(-180, 180, 1):
        for i in range(polygon.shape[0]):
            figure = polygon.copy()

            transformed = scale_figure(shape, scale)
            transformed = rotate_figure(transformed, alpha * np.pi / 180.0)

            shift = (figure[i][0] - transformed[0][0],
                     figure[i][1] - transformed[0][1])
            transformed = shift_figure(transformed, shift)
            err = compare_figures(figure, transformed)
            if err < min_err:
                min_err = err
                opt_alpha = alpha
                opt_shift = shift
    return opt_alpha, opt_shift


def find_metrics(basis, polygon):
    for i in range(len(basis)):
        # find the shape index
        if basis[i].shape[0] != polygon.shape[0]:
            continue
        index = i
        scale = int(cv.arcLength(polygon, True) / cv.arcLength(basis[i], True))
        angle, (x0, y0) = find_angle_and_shift(polygon, basis[i], scale)
        return index, int(round(x0)), int(round(y0)), scale,  int(round(angle))


def find_figures(N, basis, src_image):
    answer_list = []

    img = src_image.copy()
    img = delete_noise(img)
    shapes_points = find_shape_points_list(img)
    img2 = src_image.copy()
    img2[img2 <= 255] = 0

    for i in range(len(shapes_points)):
        hull = ConvexHull(shapes_points[i])
        for simplex in hull.simplices:
            cv.line(img2, (shapes_points[i][simplex[0]][1], shapes_points[i][simplex[0]][0]), (shapes_points[i][simplex[1]][1], shapes_points[i][simplex[1]][0]), 255, 1)

    _, contours, hierarchy = cv.findContours(img2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        perimetr = cv.arcLength(contour, True)
        polygon = np.squeeze(cv.approxPolyDP(contour, 2e-2 * perimetr, True), 1)
        index, x, y, scale, angle = find_metrics(basis, polygon)
        answer_list.append((index, x, y, scale, angle))

    #cv.drawContours(img2, contours, -1, (255, 0, 0), 1)
    #cv.imshow("Image", img2)
    #cv.waitKey(0)
    return answer_list


def main(args):
    src_file = open(args.s, "r")

    N = int(src_file.readline())
    basis = []
    for i in range(N):
        new_figure = np.array(list(map(int, src_file.readline().split(', ')))).reshape((-1, 2))
        basis.append(new_figure)

    src_image = cv.imread(args.i)
    src_image_gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)

    result = find_figures(N, basis, src_image_gray)

    all_answers = []
    print(len(result))
    for ans in result:
        answer_list = list(ans)
        all_answers.append(answer_list)
        print(*answer_list, sep=', ')
    #plot_results(all_answers, basis, src_image_gray)
    #plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initial figures')
    parser.add_argument('-s', type=str)
    parser.add_argument('-i', type=str)
    args = parser.parse_args()
    main(args)
