import argparse
import dlib
import cv2
import numpy as np
import os

from validate.utils import FaceAligner


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def predict_video(path: str, detector, face_aligner: FaceAligner,
                  classifier):
    labels = ("Live", "Spoof")
    video_capture = cv2.VideoCapture(path)

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    splited_path = path.split(".")
    splited_path[-1] = "-out.avi"

    output_path = "".join(path.split(".")[:-1]) + "-out"

    out = cv2.VideoWriter(f"{output_path}.avi",
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          25,
                          (frame_width, frame_height))
    print("Here")
    n = 0
    while video_capture.isOpened() and n < 10:
        success, frame = video_capture.read()
        n += 1
        print(n)
        if not success:
            print("File not found")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)
        if rects:
            rect = rects[0]
            aligned_face = face_aligner.align(frame, gray, rect)
            aligned_face_blob = cv2.dnn.blobFromImage(aligned_face,
                                                      scalefactor=1/255.,
                                                      size=(224, 224),
                                                      crop=True,
                                                      swapRB=True)

            aligned_face_blob = (aligned_face_blob - 0.5) / 0.5

            classifier.setInput(aligned_face_blob)
            predict = softmax(classifier.forward())
            label = labels[predict.argmax()] if predict[0][0] >= 0.9 else \
                labels[1]
            confidence = predict.max() if predict[0][0] >= 0.9 else predict[0][
                1]
            color = (0, 250, 0) if label == "Live" else (0, 0, 250)

            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        color, 4)
            cv2.putText(frame, f"{round(float(confidence), 3)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 4)

            pt1, pt2 = ((rect.left(), rect.top()), (rect.right(), rect.bottom()))
            frame = cv2.rectangle(frame, pt1, pt2, color, 2)

        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("End")
    video_capture.release()
    out.release()

    os.system(f"ffmpeg -i {output_path}.avi -an -y {output_path}.mp4")


if __name__ == '__main__':
    # Initialize
    parser = argparse.ArgumentParser(description="Pipeline for testing model by videos")

    parser.add_argument('-m',
                        '--model-path',
                        help="path to model",
                        required=True,
                        type=str)
    parser.add_argument("-p", "--shape-predictor", required=False,
                        default="resources/shape_predictor_5_face_landmarks.dat",
                        help="path to facial landmark predictor")
    parser.add_argument("-w", "--face-width", required=False,
                        default=224,
                        help="output face width")
    parser.add_argument('-d', '--device', required=False,
                        default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('VIDEOS',
                        help="one or more path to videos",
                        type=str,
                        nargs='+')

    # Parsing the argument
    args = vars(parser.parse_args())

    detector = dlib.get_frontal_face_detector() if args['device'] == 'cpu' else \
        dlib.cnn_face_detection_model_v1("resources/mmod_human_face_detector.dat")
    predictor = dlib.shape_predictor(args['shape_predictor'])
    face_aligner = FaceAligner(predictor, desiredFaceWidth=args['face_width'])
    classifier = cv2.dnn.readNetFromONNX(args['model_path'])
    for video in args['VIDEOS']:
        predict_video(video, detector, face_aligner, classifier)
