import numpy as np
import cv2
from helpers import FACIAL_LANDMARKS_68_IDXS
from helpers import FACIAL_LANDMARKS_5_IDXS
from helpers import shape_to_np


# def align_face(detections, img_rgb):
#     pad_scale = 0.8
#
#     detection = detections[0]
#     left_eye = detection["left_eye"]
#     right_eye = detection["right_eye"]
#
#     h = detection['y2'] - detection['y1']
#     w = detection['x2'] - detection['x1']
#     x = detection['x1']
#     y = detection['y1']
#     real_h = img_rgb.shape[0]
#     real_w = img_rgb.shape[1]
#
#     h_pad = h + int(h * pad_scale)
#     w_pad = w + int(w * pad_scale)
#     x_pad = x - int(w * pad_scale / 2)
#     y_pad = y - int(h * pad_scale / 2)
#
#     y1_pad = 0 if y_pad < 0 else y_pad
#     x1_pad = 0 if x_pad < 0 else x_pad
#     y2_pad = real_h if y1_pad + h_pad > real_h else y_pad + h_pad
#     x2_pad = real_w if x1_pad + w_pad > real_w else x_pad + w_pad
#
#     # img = img_rgb[detection['y1']:detection['y2'], detection['x1']:detection['x2'], :]
#     img = img_rgb[y1_pad:y2_pad, x1_pad:x2_pad, :]
#     if img.shape[0] > 0 and img.shape[1] > 0:
#         # img = alignment_procedure(img, left_eye, right_eye)
#         return img


class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        # simple hack ;)
        if len(shape) == 68:
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = (np.float32((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
                      (np.float32(leftEyeCenter[1] + rightEyeCenter[1]) // 2))
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output


if __name__ == '__main__':
    import dlib

    img = cv2.imread("resources/face.jpeg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    p = "resources/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    fa = FaceAligner(predictor=predictor)

    rects = detector(gray, 2)
    for rect in rects:
        print(rect.left(), rect.top(), rect.right(), rect.bottom())
        cv2.imshow("Aligned dlib", fa.align(img, gray, rect))
        pt1, pt2 = (rect.left(), rect.top()), (rect.right(), rect.bottom())
        cv2.imshow("Photo", cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
