import os
import glob
import random
import numpy as np

# from skimage import draw
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage import io
from skimage.transform import resize


def show(image, ax=None, rotate=False, title='', scale=1) :
    if ax is None:
        _, ax = plt.subplots(figsize=(int(scale * 20), int(scale * 20)))
    ax.axis('off')
    if rotate:
        param = (1, 0) if image.ndim == 2 else (1, 0, 2)
        image = np.transpose(image, param)[:: -1]
    ax.imshow(image)
    ax.set_title(title)
    plt.show()


def draw_bbox(ax, bbox, color=(1., 0, 1.)):
    rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor=color, facecolor='none')
    ax.add_artist(rect)


def show_collection(images, titles=[], num_rows=1, scale=1) :
    if len(titles) > 0:
        assert len(titles) == len(images)
    else:
        titles = len(images) * ['']

    fig, axes = plt.subplots(nrows=num_rows, ncols=len(images) // num_rows, figsize=(int(scale * 20), int(scale * 20)))
    for ax, image, title in zip(axes.flatten(), images, titles):
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(title)
    plt.show()

# def gray2rgb(image):
#     return np.transpose(np.stack(3 * [image]), (1, 2, 0))


# def list2table(mlist, n=20):
#     line = []
#     lines = [' | '.join(n * ['[]()']), '|'.join(n * ['-----'])]
#     k = 0
#     for w in sw:
#         line.append(w)
#         if (k + 1) % n == 0:
#             lines.append(' | '.join(line))
#             line = []
#         k += 1
#     sep = '\n{}\n'.format('|'.join(n * ['-----']))
#     text = '\n'.join(lines)
#     return text


def autolabel(rects, ax):

    ''' Attach a text label above each bar in *rects*, displaying its height.
    https://matplotlib.org/3.3.2/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    '''
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            '{:.2f}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords='offset points',
            ha='center', va='bottom'
        )


def load_simpsons_dataset(path='Simpsons-Train-Valid'):
    ''' Returns train/test splits of the Simpsons dataset. The test set comprises one single
    randomly-chosen example of each class/character. '''

    # list all valid filenames
    filenames = glob.glob(os.path.join(path, 'Train', '*'))
    filenames = [filename for filename in filenames if 'Thumbs.db' not in filename] # remove Thumbs.db

    # create the mapping
    map_character_filenames = defaultdict(list)
    for filename in filenames:
        basename = os.path.splitext(os.path.basename(filename))[0]
        character = basename[: -3] # remove the ending digits
        map_character_filenames[character].append(filename)
                                    
    train_set = []
    test_set = []
    for character in map_character_filenames:
        # select randomly 1 sample to the test set
        filename_chosen = random.choice(map_character_filenames[character])
        image = io.imread(filename_chosen)
        test_set.append((image, character))
    
        # the rest of the samples are assigned to the train set
        for filename in map_character_filenames[character]:
            if filename != filename_chosen:
                image = io.imread(filename)
                train_set.append((image, character))

    return train_set, test_set


def resize_dataset(dataset, output_shape=(200, 200)):
    ''' Resize the images in the dataset to a given output_shape. '''

    dataset_resized = []
    for image, character in dataset:
        image_resized = resize(image, output_shape)
        dataset_resized.append((image_resized, character))
    return dataset_resized


def classify_template_matching(image_query, train_set):
    min_cost = float('inf')
    label_result = ''
    image_result = None
    
    for image_candidate, label_candidate in train_set:
        cost = ((image_query - image_candidate) ** 2).sum()
        if cost < min_cost:
            min_cost = cost
            label_result = label_candidate
            image_result = image_candidate
    return image_result, label_result    