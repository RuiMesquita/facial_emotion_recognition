import tkinter as tk
import os
from PIL import ImageTk, Image
from sklearn.utils import shuffle
import numpy as np
import confusion_matrix as cm

# Data treatment
image_list = []
y_pred = []
y_true = []

root = tk.Tk()
root.geometry("500x730")
root.title('Emotion Predictions')
root.configure(background = "#495867")

systemRoot = os.getcwd()
os.chdir("./data_masks")

for dir in os.listdir():
    os.chdir("./" + dir)
    images = next(os.walk(os.getcwd()))[2]

    for image in images:
        y_true.append(dir)
        im = Image.open(image)
        im_resized = im.resize((497, 600), Image.ANTIALIAS)
        im = ImageTk.PhotoImage(im_resized)
        image_list.append(im)

    os.chdir(systemRoot + "/data_masks")

y_true, image_list = shuffle(np.array(y_true), np.array(image_list))

# Interface interaction
my_label = tk.Label(image=image_list[0])
my_label.grid(row=0, column=0, columnspan=3)


def emotion_selection(image_number, emotion):
    """
    Save the emtion selected for the given image and moves to the 
    next image on the list
    """
    global my_label
    global button_angry
    global button_disgusted
    global button_fearful
    global button_neutral
    global button_happy
    global button_sad
    global button_surprised

    y_pred.append(emotion)
    # print(emotion)

    my_label.grid_forget()
    if image_number <= len(image_list):
        my_label = tk.Label(image=image_list[image_number - 1])
        my_label.grid(row=0, column=0, columnspan=3)
    else:
        root.quit()

    button_angry = tk.Button(root, text="angry", command=lambda: emotion_selection(image_number + 1, "angry"), bg="#CE796B")
    button_disgusted = tk.Button(root, text="disgusted", command=lambda: emotion_selection(image_number + 1, "disgusted"), bg="#CE796B")
    button_fearful = tk.Button(root, text="fearful", command=lambda: emotion_selection(image_number + 1, "fearful"), bg="#CE796B")
    button_happy = tk.Button(root, text="happy", command=lambda: emotion_selection(image_number + 1, "happy"), bg="#CE796B")
    button_neutral = tk.Button(root, text="neutral", command=lambda: emotion_selection(image_number + 1, "neutral"), bg="#CE796B")
    button_sad = tk.Button(root, text="sad", command=lambda: emotion_selection(image_number + 1, "sad"), bg="#CE796B")
    button_surprised = tk.Button(root, text="surprised", command=lambda: emotion_selection(image_number + 1, "surprised"), bg="#CE796B")
    stat_bar = tk.Label(root, text = "Image %i/%i" % (image_number, len(image_list)), bd = 1, relief = tk.SUNKEN, anchor = tk.E, bg="#495867")

    button_angry.grid(row=1, column=0, sticky=tk.E + tk.W, pady = 5, padx = 2)
    button_disgusted.grid(row=1, column=1, sticky=tk.E + tk.W, pady = 5, padx = 2)
    button_fearful.grid(row=1, column=2, sticky=tk.E + tk.W, pady = 5, padx = 2)
    button_happy.grid(row = 2, column = 0, sticky=tk.E + tk.W, padx = 2)
    button_neutral.grid(row = 2, column = 1, sticky=tk.E + tk.W, padx = 2)
    button_sad.grid(row = 2, column = 2, sticky=tk.E + tk.W, padx = 2)
    button_surprised.grid(row = 3, column = 1,sticky=tk.E + tk.W, pady = 5, padx = 2)
    stat_bar.grid(row = 4, column = 0, columnspan = 3, sticky = tk.W + tk.E, pady = 5, padx = 2)



# Define all the widgets on the screen
button_angry = tk.Button(root, text="angry", command=lambda: emotion_selection(2, "angry"), bg="#CE796B")
button_disgusted = tk.Button(root, text="disgusted", command=lambda: emotion_selection(2, "disgusted"), bg="#CE796B")
button_fearful = tk.Button(root, text="fearful", command=lambda: emotion_selection(2, "fearfull"), bg="#CE796B")
button_happy = tk.Button(root, text="happy", command=lambda: emotion_selection(2, "happy"), bg="#CE796B")
button_neutral = tk.Button(root, text="neutral", command=lambda: emotion_selection(2, "neutral"), bg="#CE796B")
button_sad = tk.Button(root, text="sad", command=lambda: emotion_selection(2, "sad"), bg="#CE796B")
button_surprised = tk.Button(root, text="surprised", command=lambda: emotion_selection(2, "surprised"), bg="#CE796B")
stat_bar = tk.Label(root, text = "Image %i/%i" % (1, len(image_list)), bd = 1, relief = tk.SUNKEN, anchor = tk.E, bg="#495867")

# Placing all the widgets on the screen
button_angry.grid(row=1, column=0, sticky=tk.E + tk.W, pady = 5, padx = 2)
button_disgusted.grid(row=1, column=1, sticky=tk.E + tk.W, pady = 5, padx = 2)
button_fearful.grid(row=1, column=2, sticky=tk.E + tk.W, pady = 5, padx = 2)
button_happy.grid(row = 2, column = 0, sticky=tk.E + tk.W, padx = 2)
button_neutral.grid(row = 2, column = 1, sticky=tk.E + tk.W, padx = 2)
button_sad.grid(row = 2, column = 2, sticky=tk.E + tk.W, padx = 2)
button_surprised.grid(row = 3, column = 1,sticky=tk.E + tk.W, pady = 5, padx = 2)
stat_bar.grid(row = 4, column = 0, columnspan = 3, sticky = tk.W + tk.E, pady = 5, padx = 2)

root.mainloop()

# Once all the images have been classified the interface is closed and the confusion
# matrix is ploted with the users classification
os.chdir(systemRoot + "/output")
labels = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

for i in range(40):
    # increment the subject_number until being a valid path
    if os.path.exists("./Confusion Matrix (Subject_%i).png" % i):
        pass
    else:
       cm.plot_confusion_matrix(y_true, y_pred, labels, "Confusion Matrix (Subject_%i)" % i)
       break