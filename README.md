### Facial Emotion Recognition ###

This repository uses a deep learning neural network to do facil emotion recognition in real
time. Also, has the option to do tests for individuals using surgical masks, since the times
we are facing.

## How to use ##

1. Set the working environment to the CNN Emotions folder ("./ihmmbi/CNN Emotions")
2. Do *pip install -r requirements* for installing all the libraries used
3. The emotions.py script can be runed in three diferent mods
    * *python emotions.py --mode train* (there's already a training model avaible, so you don't need to do this part)
    * *python emotions.py --mode display* (emotion detection in real time)
    * *python emotions.py --mode matrix* (creates a confusion matrix)
    * *python emotions.py --mode maskMatrix* (creates a confusion matrix using masked faces)
4. Also can run the human_pred.py to try yourself to predict the emotions 

## More info ##

* There is also avaible the dataset_preparation.py script for dealing with a new dataset
* The dataset used was the fer2013
* Code adapted from <https://github.com/atulapra/Emotion-detection>