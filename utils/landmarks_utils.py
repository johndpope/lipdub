from custom_types import *
import cv2


def get_mask(img, shape):
    points = shape[2:15]
    # points[0] = ((shape[1] + shape[2]) / 2).astype(shape.dtype)
    # points[-1] = ((shape[15] + shape[14]) / 2).astype(shape.dtype)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int)], 255)
    nose_y = int(shape[33, 1])
    mask[:nose_y] = 0
    return mask


def get_drawer(img, shape, line_width):


    def draw_curve_(idx_list, color=(0, 255, 0), loop=False):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, line_width)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, line_width)

    return draw_curve_


def get_lips_landmarks(img, lips):
    if type(lips) is T:
        lips: T = lips.detach().cpu()
        if lips.dim() > 2:
            lips = lips.squeeze(0)
        lips = (lips + 1) * (img.shape[0] / 2)
    if lips.shape[0] > 30:
        lips = lips[48:, :]
    return lips


class Line:

    def intersect_bounds(self, target, dim: int, bounds: Optional[Tuple[float, float]]):
        if self.direction[dim] == 0:
            return None
        t = (target - self.start[dim]) / self.direction[dim]
        intersection = self.start + t * self.direction
        if bounds[0] <= intersection[1 - dim] <= bounds[1]:
            return intersection
        else:
            return None

    def intersect_x(self, target_x, bounds_y):
        return self.intersect_bounds(target_x, 0, bounds_y)

    def intersect_y(self, target_y, bounds_x):
        return self.intersect_bounds(target_y, 1, bounds_x)

    def __init__(self, points):
        self.start = points[0]
        self.direction = points[1] - points[0]


def get_intersections(pair, image) -> ARRAY:
    line = Line(pair)
    out = []
    bound = (0, image.shape[0])
    for target, dim in zip((0, 0, image.shape[0], image.shape[0]), (0, 1, 0, 1)):
        intersection = line.intersect_bounds(target, dim, bound)
        if intersection is not None:
            out.append(intersection)
        if len(out) == 2:
            break
    out = V(out)
    return out


def draw_lips_lines(image, landmarks, in_place=True):
    lips = get_lips_landmarks(image, landmarks)
    key_points = ([3, 9], [0, 6])  # height, width
    if not in_place:
        image = image.copy()
    # lips = lips.astype(np.int32)
    for i, pair in enumerate(key_points):
        points = lips[pair]
        line = get_intersections(points, image).astype(np.int32)
        if len(line) == 0:
            if i == 0:
                line = [[points[0][0], 0], [points[0][0], 256]]
            else:
                line = [[0, points[0][1]], [256, points[0][1]]]
        cv2.line(image, line[0], line[1], (0, 0, 0), 1)
    return image


def draw_lips(img, lips, scale=1):
    if type(img) is int:
        img = np.ones((img, img, 3), dtype=np.uint8) * 255
    else:
        img = img.copy()
    lips = get_lips_landmarks(img, lips)
    if scale != 1:
        lips = scale * lips + .5 * img.shape[0] * (1 - scale)
    lips = lips.astype(np.int32)
    draw_curve = get_drawer(img, lips, 1)
    draw_curve(list(range(0, 11)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(12, 19)), loop=True, color=(238, 130, 238))
    from utils import files_utils
    # for i in range(20):
    #     img_ = img.copy()
    #     cv2.circle(img_, lips[i], 2, (255, 0, 0), thickness=-1)
    #     files_utils.imshow(img_)
    #     print(i)
    return img
# 0-left out
# 3-top out
# 6-right out
# 9-bottom out
# 12-left in
# 14-top in
# 16-right in
# 18-bottom in

def vis_landmark_on_img(img, shape, line_width=2):
    shape = shape.astype(np.int32)
    draw_curve = get_drawer(img, shape, line_width)
    draw_curve(list(range(0, 16)), color=(255, 144, 25))  # jaw
    draw_curve(list(range(17, 21)), color=(50, 205, 50))  # eye brow
    draw_curve(list(range(22, 26)), color=(50, 205, 50))
    draw_curve(list(range(27, 35)), color=(208, 224, 63))  # nose
    draw_curve(list(range(36, 41)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(list(range(42, 47)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(48, 59)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(60, 67)), loop=True, color=(238, 130, 238))
    return img
