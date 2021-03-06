# import the necessary packages
import re

import numpy
import pytesseract
from pytesseract import Output

from .preprocess import *
from PIL import Image

import cv2


def getNumOnly(s):
    numeric_filter = filter(str.isdigit, s)
    numeric_string = "".join(numeric_filter)
    return numeric_string


def isIndex(item):
    return re.search("^\d{2}$", item)


def isDate(item):
    return re.search("^\d{2}\/\d{2}\/\d{4}", item)


def removeExcess(item):
    if isinstance(item, float) and numpy.isnan(item):
        return None
    elif isIndex(item):
        return item
    elif isDate(item):
        return getNumOnly(item.split('/')[0])
    else:
        splitedItem = item.split(':')
        if len(item) > 1 and len(splitedItem) > 1:
            return getNumOnly(splitedItem[0]) + ":" + getNumOnly(splitedItem[1])
        else:
            return None


def preprocessImage(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def findColumnByStr(ds, columns, cutting_str):
    data = ds.text.values
    val = 0
    for i in range(len(data)):
        if data[i] == cutting_str:
            print('found it !')
            val = 0
            for column in columns:
                val += ds[column].values[i]
            break
    return val


def getTextFromImage(img, config="--psm 12", name=True, cornersText={"top": "יום", "right": "יום", "left": "עד"},
                     deltas={"top": 0, "right": 10, "left": 0}, lang=None, atol=40):
    config = '--tessdata-dir "ocrLogic/tessdata" ' + config
    print('tesseract version', pytesseract.get_tesseract_version())
    if name:
        img = cv2.imread(img)
    # img = img[550: -300, 1200: -265]  # will change to more dynamic setting
    # cv2.imwrite('temp2.png', img)
    try:
        img = preprocessImage(img)
    except:
        print('Error In preprocess... , running without pre processing')

    d = pytesseract.image_to_data(img, output_type=Output.DATAFRAME, config=config, lang="heb")
    indexTop = findColumnByStr(d, ["top", "height"], cornersText["top"]) + deltas['top']
    indexRight = findColumnByStr(d, ["left", "width"], cornersText["right"]) + deltas['right']
    indexLeft = findColumnByStr(d, ["left", "width"], cornersText["left"]) + deltas['left']
    img = img[indexTop:, indexLeft:indexRight]  # will change to more dynamic setting
    # =========================================
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.dilate(img, None, iterations=1)
    # =========================================
    # cv2.imwrite('test1.png', img)
    d = pytesseract.image_to_data(img, output_type=Output.DATAFRAME, lang=lang, config=config)
    d = d[d.text.notnull()]
    tops = d['top'].values
    trios = []
    for i in range(len(tops)):
        # closeVals = d[d['top'].apply(np.isclose, b=tops[i], atol=40) == True]
        closeVals = d[d['top'].apply(np.isclose, b=tops[i], atol=atol) == True]
        topVals = closeVals['top'].values
        if len(closeVals[['top', 'text']]) > 0:
            # if len(closeVals['text'].values) >= 3:
            if len(closeVals['text'].values) >= 2:
                trios.append(closeVals['text'].values)
        d = d[d['top'].isin(topVals) == False]
    return processData(trios)


def cleanTrio(trio):
    rtrio = list(map(removeExcess, trio))
    rtrio = list(filter(None, rtrio))
    print(rtrio)
    return rtrio


def processData(trios):
    data = []
    latestIndex = 0
    for i in range(len(trios)):
        finalTrio = cleanTrio(trios[i])
        finalTrio = checkIndex(finalTrio, latestIndex+1)
        if len(finalTrio) == 3:
            latestIndex = int(finalTrio[2])
        data.append(finalTrio)
    return data


def checkIndex(trio, index):
    hasIndex = False
    for i in range(len(trio)):
        if isIndex(trio[i]):
            indexIndex= i
            hasIndex = True
    if not hasIndex:
        trio.append(index)
    elif indexIndex != len(trio) -1:
        indexValue = trio[indexIndex]
        trio.remove(indexValue)
        trio.append(indexValue)

    return trio


def fixListOrder(list):
    for i in range(0, len(list), 2):
        leftSide = list[i].split(':')[0]
        if int(leftSide) < 12:
            list[i], list[i + 1] = list[i + 1], list[i]

    return list


def exportToTxt(textList):
    file = open("hoursResults.txt", "w")
    text = ""
    for i in range(0, len(textList)):
        text += f"{textList[i][2]}={textList[i][1]}-{textList[i][0]}"
        if i < len(textList) - 1:
            text += "\n"

    file.write(text)
    file.close()


def listToText(textList, hasDate= True):
    text = ""
    print('text list', textList)
    for i in range(0, len(textList)):
        if hasDate:
            if len(textList[i]) == 3:
                text += f"{textList[i][2]}={textList[i][1]}-{textList[i][0]}"
                if i < len(textList) - 1:
                    text += "\n"
        else:
            if len(textList[i]) == 2:
                text += f"{textList[i][1]}-{textList[i][0]}"
                if i < len(textList) - 1:
                    text += "\n"
    return text


def convertPilToCv2(pil_image):
    pil_image = pil_image.convert('RGB')
    open_cv_image = numpy.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def cropAndExtract(imageName):
    im = Image.open(imageName)
    coordinates = getBoxes(imageName)
    rows = []
    for co in coordinates:
        row = im.crop((co[0], co[1], co[0] + co[2], co[1] + co[3])).convert('L')
        row = convertPilToCv2(row)
        d = pytesseract.image_to_data(row, output_type=Output.DICT)
        actualText = list(filter(None, d['text']))
        print(actualText)

        rows.append(row)
        cv2.imshow('img', row)
        cv2.waitKey(0)

    # width, height = im.size
    # topOffset = -1
    # rowHeight = 29
    # bottomOffset = topOffset + rowHeight
    #
    # for i in np.arange(topOffset, height - bottomOffset, rowHeight):
    #     top = i - rowHeight
    #     bottom = i
    #     left = 0
    #     right = width + 1
    #     row = im.crop((left, top, right, bottom)).convert('L')
    #
    #     d = pytesseract.image_to_data(row, output_type=Output.DICT)
    #     actualText = list(filter(None, d['text']))
    #     if len(actualText) > 0:
    #         row.show()
    #         print(actualText)

    # row = im.crop((0, topOffset, width+1, bottomOffset)).convert('L')
    # row.show()

    # d = pytesseract.image_to_data(row, output_type=Output.DICT)
    # actualText = list(filter(None, d['text']))
    # print(actualText)


def getBoxes(imageName, padding=7, show=False, filters=[]):
    img = cv2.imread(imageName)
    if filters and len(filters) > 0:
        for fil in filters:
            img = fil(img)
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 4")
    print(d['text'])
    coordinates = []
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 20:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            coordinates.append((x - padding, y - padding, w + (padding * 2), h + (padding * 2)))
            if show:
                showImg = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('img', showImg)
                cv2.waitKey(0)

    return coordinates

# if __name__ == '__main__':
#     filePath = sys.argv[1]
#     print('===========================================================================================================')
#     # imgText = getTextFromImage(filePath, config="--psm 4")
#     imgText = getTextFromImage(filePath, config="--psm 12")
#     print(len(imgText))
#     print('===========================================================================================================')
#     exportToTxt(imgText)
#     # print('===========================================================================================================')
#     # cropAndExtract(filePath)
#     # getBoxes(filePath, show=True, filters=[preprocess.biFilter, preprocess.sharpen])


