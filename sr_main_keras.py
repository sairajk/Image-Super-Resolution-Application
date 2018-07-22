from __future__ import print_function, division, absolute_import
import time
import numpy as np
import numpy.core.multiarray
import cv2
import keras
from sklearn.utils import shuffle
from keras.models import Model, load_model, model_from_json
from keras.layers.core import Lambda
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Conv2DTranspose, Input, Activation, concatenate
from skimage import io

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# path to hr and lr images
hr_dir = 'C:/Users/SAI RAJ/Desktop/HPC/floyd/super_resolution/data/processed/part1_HR'
lr_dir = 'C:/Users/SAI RAJ/Desktop/HPC/floyd/super_resolution/data/processed/part1_LR'


# load images paths from a given directory, shuffle them, and divide them into train, test and validation lists.
# Takes two directories as input, the directory for HR images and directory for lr images, and
# fraction of total data as test, default = 0.1
def load_paths(hr_direc, lr_direc, frac_as_test=0.1):
    hr_paths = sorted([os.path.join(root, file) for root, dirs, files in os.walk(hr_direc) for file in files])

    lr_paths = sorted([os.path.join(root, file) for root, dirs, files in os.walk(lr_direc) for file in files])

    hr_paths, lr_paths = shuffle(hr_paths, lr_paths)

    # shuffle and split paths
    hr_train_paths, hr_test_paths, lr_train_paths, lr_test_paths = train_test_split(hr_paths, lr_paths,
                                                                                    test_size=frac_as_test,
                                                                                    random_state=42)
    hr_train_paths, hr_val_paths, lr_train_paths, lr_val_paths = train_test_split(hr_train_paths, lr_train_paths,
                                                                                  test_size=frac_as_test,
                                                                                  random_state=42)

    return lr_train_paths, hr_train_paths, lr_test_paths, hr_test_paths, lr_val_paths, hr_val_paths


# Given a list of paths, load images for those paths. Returns 4d array as in tensorflow
# (n_images, h, w, channels =3 )
def load_data(paths):
    images = np.array([io.imread(path) for path in paths])
    if images.shape[-1] == 4:
        images = images[:, :, :, :-1]
    return images


class SuperResolution:
    def __init__(self, xtrain=None, ytrain=None, xtest=None, ytest=None, xval=None, yval=None):
        # general variables
        self.x_train = xtrain
        self.y_train = ytrain
        self.x_test = xtest
        self.y_test = ytest
        self.x_val = xval
        self.y_val = yval
        self.p_height = 240  # Height of the patch
        self.p_width = 240  # Width of the patch
        self.p_channels = 3  # Number of channels in images
        self.sf = 2         # Scaling factor, i.e by how much are we scaling the images
        self.batch_size = 16

        # Since we are using overlapping patches of size 240, the numbers below signify the corresponding
        # starting and the ending cordinates of the patches in the image
        self.start = [0, 180, 360, 540, 720, 900, 1080, 1260, 1440, 1620, 1800, 1980, 2160, 2340, 2520, 2700, 2880,
                      3060, 3240, 3420, 3600, 3780, 3960, 4140, 4320, 4500, 4680, 4860, 5040, 5220]
        self.end = [240, 420, 600, 780, 960, 1140, 1320, 1500, 1680, 1860, 2040, 2220, 2400, 2580, 2760, 2940, 3120,
                    3300, 3480, 3660, 3840, 4020, 4200, 4380, 4560, 4740, 4920, 5100, 5280, 5460]
        self.h_index = 0    # stores the maximum possible start and end index for the height of image
        self.w_index = 0    # stores the maximum possible start and end index for the width of image

        # for function preprocess
        self.iph_app = 0    # height of the padding for input
        self.ipw_app = 0    # width of the padding for input
        self.oph_app = 0    # height of the padding for output
        self.opw_app = 0    # width of the padding for output
        self.pre_flag = 0   # flag to check is preprocessing is reqd. or not

    # function to find, the number, by how many pixels do we need to pad the the images
    # to make them compatible for pre processing
    def util(self, data=None):
        _, height, width, channels = data.shape

        for j in range(len(self.start)):
            if self.start[j] < height:
                self.h_index = j
            if self.start[j] < width:
                self.w_index = j

        self.iph_app = self.end[self.h_index] - height
        self.oph_app = (self.sf * self.iph_app)

        self.ipw_app = self.end[self.w_index] - width
        self.opw_app = (self.sf * self.ipw_app)

    # function to preprocess data -- used by predict function to process raw data during prediction
    # mode is to identify if we are processing the input image or the output image -- 'lr' or 'hr'
    # appends image files with appropriate padding
    def preprocess(self, data, mode='lr'):
        self.util(data=data)

        [_, height, width, channels] = data.shape

        # To make the number of channels in all the images to 3
        if channels == 4:
            data = data[:, :, :, :-1]

        temp1 = np.array([])
        temp2 = np.array([])

        # iph_app = height of padding to be appended to the LR i/p image
        self.iph_app = self.end[self.h_index] - height

        # ipw_app = width of padding to be appended to the LR i/p image
        self.ipw_app = self.end[self.w_index] - width
        # print("append", self.iph_app, self.ipw_app, self.h_index, self.w_index)

        if mode == 'lr':
            for i in range(data.shape[0]):
                # pad image with appropritate border to make it suitable for pre procesing
                gg = cv2.copyMakeBorder(data[i], top=0, bottom=self.iph_app, left=0, right=self.ipw_app,
                                        borderType=cv2.BORDER_CONSTANT, value=[254, 254, 254])
                a, b, c = gg.shape
                gg = np.reshape(gg, newshape=(1, a, b, c))
                if i == 0:
                    temp1 = gg
                else:
                    temp1 = np.concatenate((temp1, gg), axis=0)
            return temp1

        # PRE PROCESS HR IMAGES

        # oph_app = height to be appended to the original o/p image
        self.oph_app = (self.sf * self.iph_app)

        # opw_app = width to be appended to the original o/p image
        self.opw_app = (self.sf * self.ipw_app)

        if mode == 'hr':
            for i in range(self.y_train.shape[0]):
                # pad image with appropritate border to make it suitable for pre procesing
                gg = cv2.copyMakeBorder(self.y_train[i], top=0, bottom=self.oph_app, left=0, right=self.opw_app,
                                        borderType=cv2.BORDER_CONSTANT, value=[254, 254, 254])
                a, b, c = gg.shape
                gg = np.reshape(gg, newshape=(1, a, b, c))
                if i == 0:
                    temp2 = gg
                else:
                    temp2 = np.concatenate((temp2, gg), axis=0)
            return temp2

        print("\nPre processing done.\n")

    # function to create model
    def create(self):

        # input to the model
        self.input = Input((self.p_height, self.p_width, self.p_channels), name='input')

        # Feature representation
        self.l1 = Conv2D(48, kernel_size=3, strides=1, padding='same',
                         input_shape=(self.p_height, self.p_width, self.p_channels),
                         activation='relu', name='l1')(self.input)
        # Shrinking
        self.l2 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', name='l2')(self.l1)

        # Non linear Mapping
        self.l3 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', name='l3')(self.l2)
        self.l4 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='linear', dilation_rate=2, name='l4')(self.l3)
        self.l5 = Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu', name='l5')(self.l4)

        self.l6 = Conv2D(32, kernel_size=5, strides=1, padding='same', activation='linear', dilation_rate=2, name='l6')(self.l5)
        self.l7 = Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu', name='l7')(self.l6)

        self.l8 = Conv2D(32, kernel_size=5, strides=1, padding='same', activation='linear', dilation_rate=2, name='l8')(self.l7)
        self.l9 = Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu', name='l9')(self.l8)

        self.l10 = Conv2D(32, kernel_size=5, strides=1, padding='same', activation='linear', dilation_rate=2, name='l10')(self.l9)
        self.l11 = Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu', name='l11')(self.l10)

        # Expansion
        self.l12 = Conv2D(48, kernel_size=5, strides=1, padding='same', name='l12')(self.l11)
        self.l12 = keras.layers.add([self.l1, self.l12])
        self.l12 = Activation('relu')(self.l12)

        # Image reconstruction
        self.l13 = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='linear', name='l13')(self.l12)

        # Extra at the end
        self.l14 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', name='l14')(self.l13)
        self.l15 = Conv2D(self.p_channels, kernel_size=3, strides=1, padding='same', activation='relu', name='l15')(self.l14)

        # Create model
        self.model = Model(inputs=self.input, outputs=self.l15)

        # Load Weights
        self.model.load_weights('train_best_w.h5')

        # Compile Model
        self.model.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])

        print("\nModel Created.")

    # Function to train on self.x_train data
    def train(self, epochs, batch_size):
        self.batch_size = batch_size

        # empty numpy arrays to store accuracy, loss etc.
        loss = np.array([])
        acc = np.array([])
        val_acc = np.array([])
        val_loss = np.array([])

        for e in range(epochs):
            print('epoch start :', e)
            start_time = time.clock()

            for batch in range(len(self.x_train) // batch_size):
                print('epoch:', e, 'of', epochs, "\tbatch : ", batch, 'of', len(self.x_train) // batch_size)

                # generate x training data for batch - "batch"
                x_tr_paths = self.x_train[batch * batch_size: (batch + 1) * batch_size]
                y_tr_paths = self.y_train[batch * batch_size: (batch + 1) * batch_size]

                # generate y training data for batch - "batch"
                x_tr_data = load_data(x_tr_paths)
                y_tr_data = load_data(y_tr_paths)

                history = self.model.fit(x_tr_data, y_tr_data, batch_size=batch_size, epochs=1, verbose=1)

                # save best training model according to accuracy
                if e != 0 and batch != 0:
                    if history.history['acc'] > np.max(acc):
                        self.save('train_best')

                loss = np.append(loss, history.history['loss'])
                acc = np.append(acc, history.history['acc'])

            # save model after every epoch
            self.save('final_model')

            # load validation data
            x_val_data = load_data(self.x_val)
            y_val_data = load_data(self.y_val)

            [val_l, val_a] = self.model.evaluate(x_val_data, y_val_data, batch_size=self.batch_size, verbose=0)

            print("Validation Loss :", val_l, " Validation Acc :", val_a)

            # save best validation model
            if e != 0:
                if val_a > np.max(val_acc):
                    self.save('val_best')

            val_loss = np.append(val_loss, val_l)
            val_acc = np.append(val_acc, val_a)

            print('epoch end :', e, 'time taken :', (time.clock() - start_time) // 60, 'min',
                  (time.clock() - start_time) % 60, 'sec')

        print("\nModel trained.")

    # Function to test the model on the testing data
    def test(self):
        # intial value = 0, score[0] = loss and score[1] = accuracy
        score = np.array([0, 0])

        for batch in range(len(self.x_test) // self.batch_size):
            print("test batch : ", batch, "of", len(self.x_test) // self.batch_size)

            # load x and y data for testing
            x_te_paths = self.x_test[batch * self.batch_size: (batch + 1) * self.batch_size]
            y_te_paths = self.y_test[batch * self.batch_size: (batch + 1) * self.batch_size]

            x_test_data = load_data(x_te_paths)
            y_test_data = load_data(y_te_paths)

            score = score + self.model.evaluate(x_test_data, y_test_data, batch_size=self.batch_size, verbose=0)

        score = score / (len(self.x_test) // self.batch_size)

        print("\nTest Loss :", score[0])
        print("Test Accuracy :", score[1])

    # function to predict on images, i.e. to super resolve user input images using weights of the pretrained model
    # takes only one path or image as input
    def predict(self, path_=None, image=None):
        if path_ is None and image is None:
            path_ = self.x_test[:16]
            test_img = load_data(path_)

        if path_ is not None:
            test_img = load_data(path_)
        elif image is not None:
            test_img = image
            test_img = np.expand_dims(test_img, axis=0)

        # print("test image shape", test_img.shape)

        test_img = self.preprocess(data=test_img, mode='lr')
        # print(test_img.shape)

        batch_size, height, width, channels = test_img.shape
        
        # the loop below generates overlapping patches of size self.p_height and self.p_width
        # these are then fed to the prediction network as input
        for q in range(batch_size):
            # print("predict  ", q)
            temp_x = np.array([])
            for i in range(self.h_index+1):
                for j in range(self.w_index+1):
                    gx = test_img[q, self.start[i]: self.end[i], self.start[j]: self.end[j], :]
                    gx = np.reshape(gx, (1, self.p_height, self.p_width, channels))
                    if i == 0 and j == 0:
                        temp_x = gx
                    else:
                        temp_x = np.concatenate((temp_x, gx), axis=0)

            # PREDICT
            result = self.model.predict(temp_x, batch_size=4, verbose=0)
            # print(result.shape)

            # construct the image out of prediction
            image = np.array([])
            for i in range(self.h_index+1):
                temp = np.array([])
                for j in range(self.w_index+1):
                    gg = result[int(i * (self.w_index+1) + j), :, :, :]
                    if j == 0:
                        temp = gg
                    else:
                        over_1 = temp[:, (self.sf * self.start[j]):(self.sf * self.end[j - 1]), :]
                        over_1 = over_1 + gg[:, :int(0.25 * self.sf * self.p_width), :]
                        over_1 = np.around(over_1 / 2)
                        temp[:, (self.sf * self.start[j]):(self.sf * self.end[j - 1]), :] = over_1
                        temp = np.concatenate((temp, gg[:, int(0.25 * self.sf * self.p_width):, :]), axis=1)
                    # temp.astype(np.uint8)
                if i == 0:
                    image = temp
                else:
                    over_2 = image[(self.sf * self.start[i]):(self.sf * self.end[i - 1]), :, :]
                    over_2 = over_2 + temp[:int(0.25 * self.sf * self.p_height), :, :]
                    over_2 = np.around(over_2 / 2)
                    image[(self.sf * self.start[i]):(self.sf * self.end[i - 1]), :, :] = over_2
                    image = np.concatenate((image, temp[int(0.25 * self.sf * self.p_width):, :, :]), axis=0)

            # save the image
            a, b, c = image.shape
            # print(image.shape)
            image = image[:int(a - self.oph_app), :int(b - self.opw_app), :]
            image = np.around(image)
            image = np.uint8(image)
            '''
            CODE RELATED TO SAVE IMAGES
            # print(image.shape, "image shape")
            # path = 'C:/Users/SAI RAJ/Desktop/HPC/floyd/predict/img_{0}_SRF_2_HR'.format(q) + '.png'
            # image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            # cv2.imwrite(path, image)
            '''

        print("\nPrediction Done")
        return image

    # Funtion to print model summary
    def summ(self):
        print("\n", self.model.summary(), "\n")

    # Function to save the weights
    def save(self, name):
        # name -- name of the weights, i.e train_best or validation_best etc.
        path_w = "weights/" + name + '_w.h5'
        path_m = "weights/" + name + '_m.json'

        self.model.save_weights(filepath=path_w, overwrite=True)
        
        with open(path_m, 'w') as f:
            f.write(self.model.to_json())

        print(name, "Model saved")


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val = load_paths(hr_direc=hr_dir, lr_direc=lr_dir, frac_as_test=0.1)

    run = SuperResolution(x_train, y_train, x_test, y_test, x_val, y_val)
    run.create()
    run.train(epochs=1, batch_size=5)
    run.test()
    run.predict()

