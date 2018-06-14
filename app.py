import os
from io import BytesIO

import dlib
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, jsonify, make_response
from image_gender import detect_genders

face_detector = dlib.get_frontal_face_detector()
app = Flask(__name__)


def load_image(image_url):
    try:
        if not os.path.exists(image_url):
            print("File doesn't exists")
            response = requests.get(image_url)
            image_url = BytesIO(response.content)

        img = Image.open(image_url)
        img = img.convert('RGB')
        print("Downloaded image")
        return np.array(img)
    except Exception as e:
        print("Not a valid image", e)
        return None


def get_boundary_box(boundary, image_shape):
    print(boundary)
    print(image_shape)
    X = max(boundary[3], 0)
    Y = max(boundary[0], 0)
    width = abs(X - boundary[1])
    height = abs(Y - boundary[2])
    return [X, Y, width, height]


@app.route('/', methods=['GET'])
def upload_image():
    results = {'faces': []}
    image_url = request.args.get('url')
    image = load_image(image_url)

    if image is None:
        return jsonify({'error': 'Invalid Image'})

    faces = face_detector(image, 0)
    for face in faces:
        face_boundary = face.top(), face.right(), face.bottom(), face.left()
        boundary_box = get_boundary_box(face_boundary, image.shape)
        print(boundary_box)
        gender = detect_genders(boundary_box, image)
        print(gender)
        results['faces'].append({'boundingBox': boundary_box, 'gender': gender})

    print(results)

    return make_response(jsonify(results))


if __name__ == "__main__":
    app.run("0.0.0.0", 8080)
