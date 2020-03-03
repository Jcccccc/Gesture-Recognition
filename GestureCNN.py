import time
import cv2
import numpy as np
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('th')


MODEL_FILE = 'models/gesture_weights.hdf5'

class GestureCNN:

    def __init__(self, model_path=MODEL_FILE):
        # parameters
        self.input_h, self.input_w = 52, 52
        self.input_c = 1

        self.MODEL = self.load_model(model_path)
        self.CLASSES = ['GOOD', 'SEVEN', 'PEACE', 'STOP', 'OK', 'ZERO']

        # params of prediction on frame
        (self.out_x, self.out_y) = (50, 50)
        self.color = 'rgb(255, 0, 0)' # red 
        self.font = ImageFont.truetype('font/Roboto-Light.ttf', size=32)

        #self.OUTPUT_LAYERS = [self.MODEL.getLayerNames()[i[0] - 1] for i in self.MODEL.getUnconnectedOutLayers()]
        #self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        #self.COLORS /= (np.sum(self.COLORS**2, axis=1)**0.5/255)[np.newaxis].T


    def load_model(self, model_path=None):
        model = Sequential()
    
        model.add(Conv2D(32, (3, 3), input_shape=(1, 52, 52)))
        convout1 = Activation('relu')
        model.add(convout1)
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(64, (3, 3)))
        convout2 = Activation('relu')
        model.add(convout2)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(6))
        model.add(Activation('softmax'))

        if model_path is not None:
            model.load_weights(model_path)
            print('Successfully loaded weights from '+model_path)

        layer = model.layers[15]
        self.get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])

        return model


    def preprocess(self, snap):
        snap = cv2.resize(snap, (self.input_h, self.input_w),
                          interpolation=cv2.INTER_CUBIC)
        for i in range(self.input_h):
            for j in range(self.input_w):
                (b, g, r) = snap[i][j]
                e1 = r>95 and g>40 and b>20
                e2 = max(r,g,b)-min(r,g,b)>15
                e3 = abs(int(r)-int(g))>15
                e4 = r>g and r>b
                e5 = e1 and e2 and e3 and e4
                e6 = r>220 and g>210 and b>170
                e7 = abs(int(r)-int(g))<=15
                e8 = r>b and g>b
                e9 = e6 and e7 and e8
                if (e5 or e9):
                    snap[i][j] = (255, 255, 255)
                else:
                    snap[i][j] = (0, 0, 0)
        snap = cv2.cvtColor(snap, cv2.COLOR_BGR2GRAY)
        snap = snap.reshape(1, self.input_c, self.input_h, self.input_w)
        snap = snap.astype('float32') / 255.0

        return snap


    def predict(self, snap):
        snap = self.preprocess(snap)
        #probs = self.MODEL.predict(snap)
        probs = self.get_output([snap, 0])[0][0]
        #print(probs)
        return probs


    def add_prediction(self, snap):
        snap_copy = deepcopy(snap)
        probs = self.predict(snap_copy)
        label = self.CLASSES[np.argmax(probs)]

        im_pil = Image.fromarray(snap)
        draw = ImageDraw.Draw(im_pil)
        draw.text((self.out_x, self.out_y), label,
                   fill=self.color, font=self.font)
        snap = np.asarray(im_pil)
        return snap


class VideoStreaming(object):
    def __init__(self):
        super(VideoStreaming, self).__init__()
        self.VIDEO = cv2.VideoCapture(0)

        self.MODEL = GestureCNN()

        self._preview = True
        self._detect = False
        self._exposure = self.VIDEO.get(cv2.CAP_PROP_EXPOSURE)
        self._contrast = self.VIDEO.get(cv2.CAP_PROP_CONTRAST)

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)
    
    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        self._exposure = value
        self.VIDEO.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
    
    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        self._contrast = value
        self.VIDEO.set(cv2.CAP_PROP_CONTRAST, self._contrast)

    def show(self):
        while(self.VIDEO.isOpened()):
            ret, snap = self.VIDEO.read()
            
            if ret == True:
                if self._preview:
                    # snap = cv2.resize(snap, (0, 0), fx=0.5, fy=0.5)
                    if self.detect:
                        snap = self.MODEL.add_prediction(snap)

                else:
                    snap = np.zeros((
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ), np.uint8)
                    label = 'camera disabled'
                    H, W = snap.shape
                    font = cv2.FONT_HERSHEY_PLAIN
                    color = (255, 255, 255)
                    cv2.putText(snap, label, (W//2 - 100, H//2), font, 2, color, 2)
                
                frame = cv2.imencode('.jpg', snap)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.01)

            else:
                break
        print('off')
