import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import cv2
from six.moves import urllib


def load_data(csv_file=None, save_pickle=True):
    """
    Loads csv file and extracts train/test images url and their labels

    Args
        csv_file (str) - path to dataset csv (openimages_5k.csv)
        save_pickle (bool) - saves train.p and test.p

    Returns
        x_train (list) - training image url
        y_train (list) - labels for each training images
        x_test (list) - test image url
        y_test (list) - labels for each test images
    """
    if csv_file:
        assert os.path.exists(csv_file), "Failed to locate [%s]" % (csv_file)
        df = pd.read_csv(csv_file)

        train_images = np.array(df['thumbnail_300k_url'].loc[np.where(df['subset'] == 'train')])
        train_labels = np.array(df['d_label_display_name'].loc[np.where(df['subset'] == 'train')])
        test_images = np.array(df['thumbnail_300k_url'].loc[np.where(df['subset'] == 'test')])
        test_labels = np.array(df['d_label_display_name'].loc[np.where(df['subset'] == 'test')])

        print("total train images in %s: %s" % (csv_file, len(train_images)))
        print("total test images in %s: %s" % (csv_file, len(test_images)))

        X_train, y_train = download_images(train_images, train_labels, 'train')
        X_test, y_test = download_images(test_images, test_labels, 'test')

        X_train = [cv2.imread(fname) for fname in X_train]
        X_test = [cv2.imread(fname) for fname in X_test]

        print("Save datasets as form of pickle 'train.p' and 'test.p'")
        pickle.dump({'features': X_train, 'labels': y_train}, open("train.p", "wb"))
        pickle.dump({'features': X_test, 'labels': y_test}, open("test.p", "wb"))
        np.savetxt("labels.txt", np.unique(train_labels))
    else:
        print("Loading dataset information from pickle files...")
        train = pickle.load(open("train.p", "rb"))
        test = pickle.load(open("test.p", "rb"))

        X_train, y_train = train['features'], train['labels']
        X_test, y_test = test['features'], test['labels']

    print("Dataset statistics:")
    print("total training images/labels: %s" % len(X_train))
    print("total test images/labels: %s" % len(X_test))
    print("total number of labels: %s" % len(np.unique(y_train)))
    print("labels: %s" % (np.unique(y_train)))

    return X_train, y_train, X_test, y_test

def plot_dataset_distribution(y_train):
    """
    Plots distribution of images in each class
    """
    # Count frequency of each label
    labels, counts = np.unique(y_train, return_counts=True)

    # Plot the histogram
    plt.rcParams["figure.figsize"] = [15, 5]
    axes = plt.gca()
    axes.set_xlim([-1,43])

    plt.bar(labels, counts, tick_label=labels, width=0.8, align='center')
    plt.title('Class Distribution across Training Data')
    plt.show()

def download_images(images, labels, subset='train'):
    """
    Download images from the provided images urls and put in proper folder structure

    Args
        images (list) - image urls
        labels (list) - labels to each image urls
        subset (str)  - categorized into separate folder (train, test)

    Returns
        x (list) - list of local path of images
        y (list) - list of labels of associated with each images from x
    """
    x = []
    y = []
    allLabels = list(np.unique(labels))
    for i in range(len(images)):
        download_dir = os.path.join(os.path.abspath(os.path.curdir), "dataset", subset, labels[i])
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        if images[i] != images[i] or not images[i].endswith(".jpg"):
            continue
        print("Downloading [%s] for [%s] ..." % (images[i], subset))
        try:
            if not os.path.exists(os.path.join(download_dir, os.path.basename(images[i]))):
                fpath, _ = urllib.request.urlretrieve(images[i], os.path.join(download_dir, os.path.basename(images[i])))
            else:
                fpath = os.path.join(download_dir, os.path.basename(images[i]))
        except Exception as e:
            print("Catched exception: %s" % (str(e)))
            continue

        x.append(fpath)
        y.append(allLabels.index(labels[i]))

    return x, y

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", help="openimages dataset csv")
    args = parser.parse_args()

    print(args)

    # load data from dataset
    X_train, y_train, X_test, y_test = load_data(args.csv_file)