#!/usr/bin/python3

import os
from io import BytesIO

import dlib
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
#from image_gender import detect_genders

face_detector = dlib.get_frontal_face_detector()
app = Flask(__name__)
CORS(app)


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
        print("Not a valid image found on {0}: {1}".format(image_url, e))
        return None


def get_boundary_box(boundary, image_shape):
    X = max(boundary[3], 0)
    Y = max(boundary[0], 0)
    width = min(abs(X - boundary[1]), image_shape[0])
    height = min(abs(Y - boundary[2]), image_shape[1])
    return [X, Y, width, height]


@app.route('/', methods=['GET'])
def health_check():
    return "Yay !! It's working!!"


@app.route('/', methods=['POST'])
@cross_origin()
def image_handler():
    results = {'faces': []}
    if request.json is not None:
        image_url = request.json['url']
    elif request.form is not None:
        image_url = request.form['url']
    else:
        return "You must pass 'url' parameter."

    print("Received image url: {0}".format(image_url))
    image = load_image(image_url)

    if image is None:
        return jsonify({'error': 'Invalid Image'})

    faces = face_detector(image, 1)
    for face in faces:
        face_boundary = face.top(), face.right(), face.bottom(), face.left()
        boundary_box = get_boundary_box(face_boundary, image.shape)
        #gender = detect_genders(boundary_box, image)
        results['faces'].append({'boundingBox': boundary_box})

    print(results)
    return jsonify(results)


if __name__ == "__main__":
    app.run("0.0.0.0", 8080)
