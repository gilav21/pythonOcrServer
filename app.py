import os

from flask import Flask, request
from ocrLogic.ocrProvider import getTextFromImage, listToText
from flask_cors import CORS
import numpy as np
import cv2

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/getTextFromImagePath', methods=['POST'])
def getTextFromImagePath():
    # image = request.files['file'];
    # convert string of image data to uint8
    # nparr = np.fromstring(request.data, np.uint8)
    nparr = np.frombuffer(request.data, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('hours.png', image)
    textList = getTextFromImage('hours.png', name=True)
    text = listToText(textList)
    os.remove('hours.png')
    return text


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, port=port)
