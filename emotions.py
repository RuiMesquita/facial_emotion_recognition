import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import os
import tqdm
import imageio
from sklearn.metrics import confusion_matrix
import confusion_matrix as cm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Removes warning messages
cv2.ocl.setUseOpenCL(False)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display/matrix/maskMatrix")
mode = ap.parse_args().mode


# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1), len(model_history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    os.chdir("./output")
    fig.savefig('plot.png')
    plt.show()


def emotion_recog(frame):
    """
    Predicts the emotion given one of the data/test images
    """
    emotion_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}

    # need to convert (48, 48, 3) -> (1, 48, 48, 1)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_dim = np.expand_dims(np.expand_dims(gray_image, -1), 0)
    #print(processed_dim.shape)

    prediction = model.predict(processed_dim)
    maxindex = int(np.argmax(prediction))

    # Print the alghoritm prediction
    return emotion_dict[maxindex]


def emotion_recog_with_mask(frame, facecasc):
    """
    Predicts the emotion once given one image and a cascade classifier
    """
    emotion_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return "None"
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 255), 3)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return emotion_dict[maxindex]


# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'
num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# train option using -mode train
if mode == "train":
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    plot_model_history(model_info)
    model.save_weights('model.h5')

# display option using -mode display (real time emotion detection using pc camera)
elif mode == "display":
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 0, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

elif mode == "matrix":

    y_pred = []
    y_true = []

    # Load the trained model
    model.load_weights('model.h5')
    w_env = os.getcwd()
    
    # changing to the test directory
    os.chdir("./data/test")

    # Saving a copy of the test path
    original = os.getcwd()

    # Going through all the directories inside the test directory
    for dir in os.listdir():
        os.chdir("./" + dir)
        images = next(os.walk(os.getcwd()))[2]

        for image in images:
            y_true.append(dir)
            frame = cv2.imread(image)
            frame = np.asarray(frame)
            em_pred = emotion_recog(frame)
            y_pred.append(em_pred)

        # go down one directory
        os.chdir(original)
    
    # checking list lengths 
    print(len(y_pred))
    print(len(y_true))

    labels = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
    
    # Return to the main environment to save the plot
    os.chdir(w_env + "/output")
    cm.plot_confusion_matrix(y_true, y_pred, labels, "Confusion Matrix (No Mask)")

elif mode == "maskMatrix":
    y_pred = []
    y_true = []

    # Load the trained model
    model.load_weights('model.h5')

    # Load the cascade classifier
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    w_env = os.getcwd()
    
    # changing to the test directory
    os.chdir("./data_masks")

    # Saving a copy of the test path
    original = os.getcwd()

    # Going through all the directories inside the test directory
    for dir in os.listdir():
        os.chdir("./" + dir)
        images = next(os.walk(os.getcwd()))[2]

        for image in images:
            y_true.append(dir)
            frame = cv2.imread(image)
            frame = np.asarray(frame)
            em_pred = emotion_recog_with_mask(frame, facecasc)
            y_pred.append(em_pred)

        # go down one directory
        os.chdir(original)
    
    print("pred", y_pred)
    print("true", y_true)

    labels = ["None", "Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
    
    # Return to the main environment to save the plot
    os.chdir(w_env + "/output")

    # Plot confusion matrix
    cm.plot_confusion_matrix(y_true, y_pred, labels, "Confusion Matrix (wMasks)")
