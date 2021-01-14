import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# convert string to integer
def atoi(s):
    """
    Converts strings to integers
    """
    n = 0
    for i in s:
        n = n * 10 + ord(i) - ord("0")
    return n


def plotBarGraph(labels, n_train, n_test):
    """
    Plots a bar graph with all the dataset information
    """
    plt.subplot(121)
    plt.bar(labels, n_train, color=lc)
    plt.title("Treino")
    plt.ylabel("Nº de imagens")
    plt.xticks(rotation=45, ha="right")

    plt.subplot(122)
    plt.bar(labels, n_test, color=lc)
    plt.title("Teste")
    plt.xticks(rotation=45, ha="right")

    os.chdir("./output")
    plt.savefig("Emotions_distribution.png", bbox_inches='tight')
    plt.show()


# making folders
outer_names = ['test', 'train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
lc = ['#99DDC8', '#95BF74', '#659B5E', '#556F44', '#283F3B', '#3C504D', '#4E605D']

os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('data', outer_name), exist_ok=True)
    for inner_name in inner_names:
        os.makedirs(os.path.join('data', outer_name, inner_name), exist_ok=True)

# keeping count of each category
angry = 0
disgusted = 0
fearful = 0
happy = 0
sad = 0
surprised = 0
neutral = 0
angry_test = 0
disgusted_test = 0
fearful_test = 0
happy_test = 0
sad_test = 0
surprised_test = 0
neutral_test = 0

# store csv content into a dataframe
df = pd.read_csv('./fer2013.csv')
mat = np.zeros((48, 48), dtype=np.uint8)
print("Saving images...")

# read the csv file line by line
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()

    # image size is 48x48 pixels
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    # images for training
    if i < 28709:
        if df['emotion'][i] == 0:
            img.save('data/train/angry/im' + str(angry) + '.png')
            angry += 1
        elif df['emotion'][i] == 1:
            img.save('data/train/disgusted/im' + str(disgusted) + '.png')
            disgusted += 1
        elif df['emotion'][i] == 2:
            img.save('data/train/fearful/im' + str(fearful) + '.png')
            fearful += 1
        elif df['emotion'][i] == 3:
            img.save('data/train/happy/im' + str(happy) + '.png')
            happy += 1
        elif df['emotion'][i] == 4:
            img.save('data/train/sad/im' + str(sad) + '.png')
            sad += 1
        elif df['emotion'][i] == 5:
            img.save('data/train/surprised/im' + str(surprised) + '.png')
            surprised += 1
        elif df['emotion'][i] == 6:
            img.save('data/train/neutral/im' + str(neutral) + '.png')
            neutral += 1

    # images for testing
    else:
        if df['emotion'][i] == 0:
            img.save('data/test/angry/im' + str(angry_test) + '.png')
            angry_test += 1
        elif df['emotion'][i] == 1:
            img.save('data/test/disgusted/im' + str(disgusted_test) + '.png')
            disgusted_test += 1
        elif df['emotion'][i] == 2:
            img.save('data/test/fearful/im' + str(fearful_test) + '.png')
            fearful_test += 1
        elif df['emotion'][i] == 3:
            img.save('data/test/happy/im' + str(happy_test) + '.png')
            happy_test += 1
        elif df['emotion'][i] == 4:
            img.save('data/test/sad/im' + str(sad_test) + '.png')
            sad_test += 1
        elif df['emotion'][i] == 5:
            img.save('data/test/surprised/im' + str(surprised_test) + '.png')
            surprised_test += 1
        elif df['emotion'][i] == 6:
            img.save('data/test/neutral/im' + str(neutral_test) + '.png')
            neutral_test += 1

train_emotions = [angry, disgusted, fearful, happy, sad, surprised, neutral]
test_emotions = [angry_test, disgusted_test, fearful_test, happy_test, sad_test, surprised_test, neutral_test]
plotBarGraph(inner_names, train_emotions, test_emotions)

print("Done!")
