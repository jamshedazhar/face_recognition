import os
from io import BytesIO

import dlib
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, jsonify
from image_gender import detect_genders

face_detector = dlib.get_frontal_face_detector()
app = Flask(__name__)


def load_image(image_url):
    try:
        if not os.path.exists(image_url):
            response = requests.get(image_url)
            image_url = BytesIO(response.content)
            print('File downloaded from URL: {0}'.format(image_url))

        img = Image.open(image_url)
        img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print("Not a valid image", e)
        return None


def get_boundary_box(boundary, image_shape):
    X = max(boundary[3], 0)
    Y = max(boundary[0], 0)
    width = min(abs(X - boundary[1]), image_shape[0])
    height = min(abs(Y - boundary[2]), image_shape[1])
    return [X, Y, width, height]


@app.route('/', methods=['GET'])
def image_handler():
    results = {'faces': []}
    image_url = request.args.get('url')
    image = load_image(image_url)

    if image is None:
        return jsonify({'error': 'Invalid Image'})

    faces = face_detector(image, 0)
    for face in faces:
        face_boundary = face.top(), face.right(), face.bottom(), face.left()
        boundary_box = get_boundary_box(face_boundary, image.shape)
        gender = detect_genders(boundary_box, image)
        results['faces'].append({'boundingBox': boundary_box, 'gender': gender})

    print(results)

    return jsonify(results)


if __name__ == "__main__":
    app.run("0.0.0.0", 8080)
