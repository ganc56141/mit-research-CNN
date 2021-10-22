
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, InputLayer, BatchNormalization
from tensorflow.keras import utils
import os, pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # compiler optimization flag
path = os.path.dirname(os.path.realpath(__file__))

def preprocess(img_path, dim):
    # load in the image
    image = load_img(img_path, target_size=(dim, dim, 3))
    # image.show()
    
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    
    # reshape (number of samples, rows, columns, channels)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    # quick preprocess (simply noramlize to the mean of the 3 channels)
    image = preprocess_input(image)
    return image



def cancer_CNN_VGG19():
    dim = 224
    batch_size = 100
    model = VGG19(include_top=False, input_shape=(dim, dim, 3))
    print(model.summary())
    
    imG = ImageDataGenerator()
    
    X_train = imG.flow_from_directory(f"{path}/colon_training_set/train/", 
                                      class_mode="categorical", 
                                      shuffle=False, 
                                      batch_size=batch_size, 
                                      target_size=(224, 224))
    X_test = imG.flow_from_directory(f"{path}/colon_training_set/test/", 
                                     class_mode="categorical", 
                                     shuffle=False, 
                                     batch_size=batch_size, 
                                     target_size=(224, 224))

    y_train = utils.to_categorical(X_train.labels)
    y_test = utils.to_categorical(X_test.labels)
    
    # convolution using pretrained kernels
    
    # X_convolved_train = model.predict(X_train)
    # X_convolved_test = model.predict(X_test)
    
    # pickle.dump(X_convolved_train, open(f'{path}/Xconvolvedtrain.pickle', 'wb') )
    # pickle.dump(X_convolved_test, open(f'{path}/Xconvolvedtest.pickle', 'wb'))
    
    X_convolved_train = pickle.load(open(f'{path}/Xconvolvedtrain.pickle', 'rb'))
    X_convolved_test = pickle.load(open(f'{path}/Xconvolvedtest.pickle', 'rb'))
    print(len(X_convolved_test))
    print(len(y_test))
    
    
    classifier_model = Sequential()
    classifier_model.add(Flatten(input_shape=(7,7,512)))        # connect with previous MaxPoolingLayer
    classifier_model.add(Dense(100, activation='relu'))
    classifier_model.add(Dropout(0.4))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Dense(2, activation='softmax'))
    
    # compile the model
    classifier_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    classifier_model.summary()
    
    
    # train model using features generated from VGG16 model
    weights_path = f'{path}/cancer_VGG19_weights.h5'
    if not os.path.exists(weights_path):
        classifier_model.fit(x = X_convolved_train, 
                            y = y_train, 
                            epochs = 10, 
                            batch_size = batch_size, 
                            validation_data = (X_convolved_test, y_test))
        classifier_model.save_weights(weights_path)
    else:
        ans = str(input("Do you want to overwrite?"))
        if ans == '1':
            classifier_model.fit(x = X_convolved_train, 
                            y = y_train, 
                            epochs = 20, 
                            batch_size = batch_size, 
                            validation_data = (X_convolved_test, y_test))
            classifier_model.save_weights(weights_path)
        else:
            classifier_model.load_weights(weights_path)
    
    
    # evaluate performance
    _, train_acc = classifier_model.evaluate(x=X_convolved_train, y=y_train)
    _, test_acc = classifier_model.evaluate(x=X_convolved_test, y=y_test)
    print("train acc", train_acc)
    print("test acc", test_acc)
  


def cancer_CNN_V2():
    dim = 224
    batch_size = 10
    
    imG = ImageDataGenerator()
    
    X_train = imG.flow_from_directory(f"{path}/colon_training_set/train/", 
                                      class_mode="categorical", 
                                      shuffle=True, 
                                      batch_size=batch_size, 
                                      target_size=(224, 224))
    X_test = imG.flow_from_directory(f"{path}/colon_training_set/test/", 
                                     class_mode="categorical", 
                                     shuffle=True, 
                                     batch_size=batch_size, 
                                     target_size=(224, 224))

    y_train = utils.to_categorical(X_train.labels)
    y_test = utils.to_categorical(X_test.labels)
    
    
    
    # build a sequential model
    model = Sequential()
    model.add(InputLayer(input_shape=(224, 224, 3)))

    # 1st conv block
    model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    # 2nd conv block
    model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    # 3rd conv block
    model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    # ANN block
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.25))
    # output layer
    model.add(Dense(units=2, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    print(model.summary())
    
    # fit on data for 30 epochs
    # model.fit(X_train, epochs=10, validation_data=X_test)
    # model.save_weights(f'{path}/cancer_weights.h5')
    
    model.load_weights(f'{path}/cancer_weights.h5')




import base64
def convert_to_jpg(imgstring):
    imgdata = base64.b64decode(imgstring)
    filename = f'{path}/PATIENT_IMAGE.jpeg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
    return filename

    
def predict_using_CNN(img_path):
    dim = 224
    batch_size = 100
    model = VGG19(include_top=False, input_shape=(dim, dim, 3))
    
    image = preprocess(img_path, dim)
    stage_1_conv = model.predict(image)
    
    classifier_model = Sequential()
    classifier_model.add(Flatten(input_shape=(7,7,512)))        # connect with previous MaxPoolingLayer
    classifier_model.add(Dense(100, activation='relu'))
    classifier_model.add(Dropout(0.4))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Dense(2, activation='softmax'))
    
    classifier_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    
    weights_path = f'{path}/cancer_VGG19_weights.h5'
    classifier_model.load_weights(weights_path)
    
    x = classifier_model.predict(stage_1_conv)
    return 'positive' if x.argmax(axis=1)[0] == 1 else 'negative'
    
    
    
    

    
if __name__ == '__main__':
    cancer_CNN_VGG19()
    # imgstring = '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'
    # img_path = convert_to_jpg(imgstring)
    # res = predict_using_CNN(img_path)
    # print(res)
    
    # cancer_CNN_V2()
    
    