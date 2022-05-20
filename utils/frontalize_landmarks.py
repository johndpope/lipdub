from custom_types import *
import constants
import math


def get_eye_centers(landmarks):
    # Given a numpy array of [68,2] facial landmarks, returns the eye centers
    # of a face. Assumes the DLIB landmark scheme.

    landmarks_eye_left = landmarks[36:42, :]
    landmarks_eye_right = landmarks[42:48, :]

    center_eye_left = np.mean(landmarks_eye_left, axis=0)
    center_eye_right = np.mean(landmarks_eye_right, axis=0)

    return center_eye_left, center_eye_right


def get_procrustes(
        landmarks,
        translate=True,
        scale=True,
        rotate=True,
        template_landmarks=None):


    landmarks_standard = landmarks.copy()

    # translation
    if translate is True:
        landmark_mean = np.mean(landmarks, axis=0)
        landmarks_standard = landmarks_standard - landmark_mean

    # scale
    if scale is True:
        landmark_scale = math.sqrt(
            np.mean(np.sum(landmarks_standard ** 2, axis=1))
        )
        landmarks_standard = landmarks_standard / landmark_scale

    if rotate is True:
        # rotation
        center_eye_left, center_eye_right = get_eye_centers(landmarks_standard)

        # distance between the eyes
        dx = center_eye_right[0] - center_eye_left[0]
        dy = center_eye_right[1] - center_eye_left[1]

        if dx != 0:
            f = dy / dx
            a = math.atan(f)  # rotation angle in radians
            # ad = math.degrees(a)
            # print('Eye2eye angle=', ad)

        R = np.array([
            [math.cos(a), -math.sin(a)],
            [math.sin(a), math.cos(a)]
        ])  # rotation matrix
        landmarks_standard = np.matmul(landmarks_standard, R)

    '''
    adjusting facial parts to a tamplate face
    displacing face parts to predetermined positions (as defined by the 
    template_landmarks), except from the eyebrows, which convey important 
    expression information attention! this only makes sense for frontal faces!
    '''
    if template_landmarks is not None:
        # mouth
        anchorpoint_template = np.mean(template_landmarks[50:53, :], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[50:53, :], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[48:, :] += displacement

        # right eye
        anchorpoint_template = np.mean(template_landmarks[42:48, :], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[42:48, :], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[42:48, :] += displacement
        # right eyebrow (same displaycement as the right eye)
        landmarks_standard[22:27, :] += displacement  # TODO: only X?

        # left eye
        anchorpoint_template = np.mean(template_landmarks[36:42, :], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[36:42, :], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[36:42, :] += displacement
        # left eyebrow (same displaycement as the left eye)
        landmarks_standard[17:22, :] += displacement  # TODO: only X?

        # nose
        anchorpoint_template = np.mean(template_landmarks[27:36, :], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[27:36, :], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[27:36, :] += displacement

        # jaw
        anchorpoint_template = np.mean(template_landmarks[:17, :], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[:17, :], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[:17, :] += displacement

    return landmarks_standard, landmark_scale, landmark_mean


def get_landmark_matrix(ls_coord):
    mid = len(ls_coord) // 2
    landmarks = np.array([ls_coord[:mid], ls_coord[mid:]])
    return landmarks.T


class FrontalizeLandmarks:

    def __call__(self, landmarks):
        landmarks_standard, scale, center = get_procrustes(landmarks, template_landmarks=None)
        landmark_vector = np.hstack(
            (landmarks_standard[:, 0].T, landmarks_standard[:, 1].T, 1)
        )  # add interception
        landmarks_frontal = np.matmul(landmark_vector, self.frontalization_weights)
        landmarks_frontal = get_landmark_matrix(landmarks_frontal)
        landmarks_frontal = landmarks_frontal * scale + center
        return landmarks_frontal

    def __init__(self):
        self.frontalization_weights = np.load(f'{constants.PROJECT_ROOT}/weights/frontalization_weights.npy')
