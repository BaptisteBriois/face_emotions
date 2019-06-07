from statistics import mode

import cv2
import sys
import base64
import time
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.io import imsave
from PIL import Image

# parameters for loading data and images
detection_model_path = 'trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

def xdog(im, gamma=0.99, phi=200, eps=-0.1, k=1.6, sigma=0.8, binarize=False):
    # Source : https://github.com/CemalUnal/XDoG-Filter
    # Reference : XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization
    # Link : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.151&rep=rep1&type=pdf
    imf1 = gaussian_filter(im, sigma)
    imf2 = gaussian_filter(im, sigma * k)
    imdiff = imf1 - gamma * imf2
    imdiff = (imdiff < eps) * 1.0  + (imdiff >= eps) * (1.0 + np.tanh(phi * imdiff))
    imdiff -= imdiff.min()
    imdiff /= imdiff.max()
    if binarize:
        th = threshold_otsu(imdiff)
        imdiff = imdiff >= th
    imdiff = imdiff.astype('float32')
    return imdiff

def main(image):
    encoded_data = image
    imageBase64 = encoded_data.split(',')[1]  # Just get index with base64  image data // Caution in react you have three option of picture forrmat
    # imageBase64 += "=" * ((4 - len(imageBase64) % 4) % 4)
    with open("pictures/original/original.png", 'wb') as f:
        f.write(base64.b64decode(imageBase64))
    bgr_image = cv2.imread("pictures/original/original.png")
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:
        ts = str(time.time())
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        """
            img = np.array()
            img.shape -> (w, h, c)

            n = 10
            xs = np.random.randint(0, w, n)
            ys = ---------------------h----

            HINT = np.zeros((w, h, 4), dtype=np.uint8)

            for pixels:
                HINT[xi, yi, :3] = img[xi, yi, :3]
                HINT[xi, yi, 3]  = 255

        """

        x, y, w, h = face_coordinates
        roi_color = bgr_image[y:y + h, x:x + w]
        c = roi_color.shape[2]
        # n = 10
        # xs = np.random.randint(w, size=n)
        # ys = np.random.randint(h, size=n)

        print("[INFO] Object found. Saving locally.")
        cv2.imwrite('pictures/face/' + ts + '_faces.png', roi_color)
        im = cv2.imread('pictures/face/' + ts + '_faces.png')
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        xdogim = xdog(imgray, gamma=0.95, phi=230, eps=-0.1, k=1.6, sigma=1, binarize=True)
        encoded_xdog = xdogim
        imsave('pictures/contour/' + ts + '_faces_grey.png', xdogim)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            face_color = [0, 0, 255]
        elif emotion_text == 'sad':
            face_color = [255, 0, 0]
        elif emotion_text == 'happy':
            face_color = [0, 255, 255]
        elif emotion_text == 'surprise':
            face_color = [0, 140, 255]
        else:
            face_color = [0, 255, 0]

        xs = [w/5, w/5, w/5, w*3/5, w*3/5, w/2, w*9/10, w*9/10]
        ys = [h/3, h*2/3, h/2, h/4, h*3/4, h/2, h*2/5, h*3/5]

        hint = np.zeros((w, h, 4), dtype=np.uint8)

        for i in range(0, 8):
            x, y = xs[i], ys[i]
            hint[
                [x + _x for _x in list(range(3)) * 3 if (x + _x) > -1 and (x + _x) < w],
                [y + _y for _y in (0, ) * 3 + (1, ) * 3 + (2, ) * 3 if (y + _y) > -1 and (y + _y) < h],
                :3
            ] = face_color

            hint[
                [x + _x for _x in list(range(3)) * 3 if (x + _x) > -1 and (x + _x) < w],
                [y + _y for _y in (0, ) * 3 + (1, ) * 3 + (2, ) * 3 if (y + _y) > -1 and (y + _y) < h],
                3
            ] = 255

        cv2.imwrite('pictures/hint/' + ts + '_hint.png', hint)

        with open("pictures/contour/" + ts + "_faces_grey.png", 'rb') as f:
            xdogContour = f.read()

        with open("pictures/hint/" + ts + "_hint.png", 'rb') as f:
            hintread = f.read()

        

        return {'sketch': base64.b64encode(xdogContour), 'hint': base64.b64encode(hintread)}
