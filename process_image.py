import keras_ocr
import cv2
import numpy as np
import math
import os

# Cropping RoI
def crop_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    close = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    dilate = cv2.dilate(close, dilate_kernel, iterations=1)

    cnts, _ = cv2.findContours(dilate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts]

    c_ = cnts[np.argmax(areas)]
    x, y, w, h = cv2.boundingRect(c_)
    return img[y:y+h, x:x+w]

# Remove text overlays
def midpoint(x1, y1, x2, y2):
    return (int((x1 + x2)/2), int((y1 + y2)/2))

def inpaint_text(img, pipeline):
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)

    return cv2.inpaint(img, mask, 2, cv2.INPAINT_TELEA)

# Remove 'P' letter artifacts
def inpaint_P(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.dilate(close, dilate_kernel, iterations=1)

    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        area = cv2.contourArea(c)
        if 800 < area < 900:
            x, y, w, h = cv2.boundingRect(c)
            img[y:y+h, x:x+w] = 0
    return img

# Remove line artifacts from white dots
def inpaint_dots(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    thresholded = np.logical_and(*[lab[..., i] > t for i, t in enumerate([200, 0, 0])])
    thresholded = thresholded.astype('uint8') * 255

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    close = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    dilate = cv2.dilate(close, dilate_kernel, iterations=1)

    mask = np.zeros(img.shape[:2], dtype='uint8')
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    listx, listy, listw, listh = [], [], [], []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        listx.append(x)
        listy.append(y)
        listw.append(w)
        listh.append(h)

    roundedx = np.floor(np.array(listx) / 10) * 10
    counts = Counter(roundedx)
    indxs = np.where(roundedx == counts.most_common(1)[0][0])[0]

    if len(indxs) > 8:
        for x_, y_, w_, h_ in zip(np.array(listx)[indxs], np.array(listy)[indxs], np.array(listw)[indxs], np.array(listh)[indxs]):
            mask[y_:y_+h_, x_:x_+w_] = 255

        img = cv2.inpaint(img, mask, 4, cv2.INPAINT_TELEA)
    return img

# Main processing function
def preprocess_image(orig_image):
    pipeline = keras_ocr.pipeline.Pipeline()
    img = crop_image(orig_image)
    img = inpaint_P(img)
    img = inpaint_dots(img)
    img = inpaint_text(img, pipeline)
    return img

if __name__ == "__main__":
    orig_image = cv2.imread('/path/to/dataset')
    result = preprocess_image(orig_image)
    cv2.imwrite('-----------', result)

