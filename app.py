import os

from flask import Flask, request
from ocrLogic.ocrProvider import getTextFromImage, listToText
from flask_cors import CORS
import numpy as np
import cv2
import hashlib

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

types = {
    "sap": {
        "corners": {"top": "יום", "right": "יום", "left": "עד"},
        "deltas": {"top": 0, "right": 10, "left": 0},
        "atol": 40,
        "hasDate": True
    },
    "alt1": {
        "corners": {"top": "תאריך", "right": "תאריך", "left": 'סה"כ'},
        "deltas": {"top": 10, "right": 12, "left": 25},
        "atol": 35,
        "hasDate": True
    }
}


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/getTextFromImagePath', methods=['POST'])
def getTextFromImagePath():
    image = request.files['files']
    fileType = request.form['type']
    fileParts = image.filename.split('.')
    nparr = np.fromfile(image, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    fileName = 'hours.' + fileParts[len(fileParts) - 1]
    cv2.imwrite(fileName, image)
    print('hash', md5(fileName))
    textList = getTextFromImage(fileName, name=True, cornersText=types[fileType]['corners'],
                                deltas=types[fileType]['deltas'], atol=types[fileType]['atol'])
    # textList = getTextFromImage('05.22.jpeg', name=True, cornersText=types['alt1']['corners'],
    #                             deltas=types['alt1']['deltas'])
    text = listToText(textList, hasDate=types[fileType]['hasDate'])
    # os.remove('hours.png')
    return text


def fitFormat(imageName, corners={"top": "יום", "right": "יום", "left": "עד"},
              deltas={"top": 0, "right": 0, "left": 0}):
    # deltas = {"top": 15, "right": 12, "left": 25}
    max = 0
    results = {"top": 0, "right": 0, "left": 0}
    for tops in range(0, 30, 5):
        for rights in range(0, 30, 5):
            for lefts in range(0, 30, 5):
                tempDeltas = {"top": deltas['top'] + tops, "right": deltas['right'] + rights,
                              "left": deltas['left'] + lefts}
                textList = getTextFromImage(imageName, name=True, cornersText=corners, deltas=tempDeltas)
                filteredArr = list(filter(None, textList))
                if len(filteredArr) > max:
                    max = len(filteredArr)
                    results['top'] = tops
                    results['right'] = rights
                    results['left'] = lefts

    return results


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, port=port)
