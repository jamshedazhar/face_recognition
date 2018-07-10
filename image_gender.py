import cv2
import numpy as np
from keras.models import load_model

gender_model_path = '/home/ubuntu/face_recognition/trained_models/gender_models/simple_CNN.81-0.96.hdf5'
gender_classifier = load_model(gender_model_path, compile=False)

# https://github.com/keras-team/keras/issues/6462
gender_classifier._make_predict_function()

gender_target_size = gender_classifier.input_shape[1:3]
gender_offsets = (10, 10)


def detect_genders(face, image):
    result = ""
    x1, x2, y1, y2 = apply_offsets(face, gender_offsets)
    rgb_face = image[y1:y2, x1:x2]

    try:
        rgb_face = cv2.resize(rgb_face, gender_target_size)
    except Exception as e:
        print('Error while resizing the image', e)
        return result

    rgb_face = pre_process_input(rgb_face)
    rgb_face = np.expand_dims(rgb_face, 0)
    gender_prediction = gender_classifier.predict(rgb_face)
    result = {
        'woman': float(gender_prediction[0][0]),
        'man': float(gender_prediction[0][1])
    }

    return result


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return max(x - x_off, 0), x + width + x_off, max(y - y_off, 0), y + height + y_off


def pre_process_input(x, v2=False):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
