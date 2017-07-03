import numpy as np
import os
import random

from data import helpers_image, helpers


###################################################################
#############################################################################
def load_image_list(list_file_path, img_dir=None, delimeter=None):
    """
    Read a label file with each line containing a path to an image and its integer class label, separated by a space
    the image path should contain no space, if so, it'll be surrounded by double wuotes which have to be accounted for when reading the file

    :param list_file_path: abs path to a labels file
    :param img_dir: optiona rppt path to be appended before the images path in case the images paths are relative
    :return: list of images path and absolute path and a list of corresponding labels
    """

    f = open(list_file_path, 'r')
    image_fullpath_list = []
    image_list = []
    labels = []
    for line in f:
        if delimeter:
            items = line.split(delimeter)
        else:
            items = line.split()
        nItems = len(items)
        lbl = items[-1]
        impath = None
        if nItems==2:
            impath = items[0].strip()
        else:
            items2 = line.split('"')
            if len(items2)>3:
                raise (line + 'Error, cant read file, many items found in line')
            impath = items2[1]

        if not impath:
            raise ('Error, cant read file')

        if img_dir:  # a root image directory is passed, meaning paths written in the file are relative to that directory
            image_fullpath_list.append(os.path.join(img_dir, impath))
            image_list.append(impath)
        else:
            image_fullpath_list.append(impath)

        labels.append(lbl.strip())
    return image_fullpath_list, labels, image_list


###################################################################
import csv
def Generate_LabelsFile(imagesRootDir, trainF, valF, categoriesFile, validationPercent=0.2):
    """
    Given the root images directory, generate labels text files for the training and validation files.
    Each line in a labels file contain a path to an image and it groundtruth label as integer.
    Also saves a categories file with each line representing a class name for the corresponding integer labels

    :param validationPercent: float representing percent of images to take from that root directory into the validation set, Default=0.2
    :return: paths to the saved labels files
    """

    categoriesDirs = os.listdir(imagesRootDir)
    with open(categoriesFile, 'w') as lFile:
        for item in categoriesDirs:
            lFile.write("%s\n" % item)

    with open(trainF, 'wb') as tF, open(valF, 'w') as vF:
        tF_wr = csv.writer(tF, delimiter=' ')
        vF_wr = csv.writer(vF, delimiter=' ')
        for label, category in enumerate(categoriesDirs):
            files = os.listdir(imagesRootDir + '/' + category)
            nAllImages = len(files)
            nTrain = nAllImages - int(round(nAllImages * validationPercent))
            trainImgs = files[0:nTrain]
            valFiles = files[nTrain:]

            for img in trainImgs:
                tF_wr.writerow([imagesRootDir + '/' + category + '/' + img] + [label])
            for img in valFiles:
                vF_wr.writerow([imagesRootDir + '/' + category + '/' + img] + [label])

    return categoriesDirs

###################################################################
def read_image_files_from_csv(csv_file, data_dir=None, labels_file=None, test=False, delimiter=None):
    """Build a list of all images files and labels in the data set.

    Args:
      data_dir: string, path to the root directory of images.

      labels_file: string, path to the labels file.


    Returns:
      filenames: list of strings; each string is a path to an image file.
      texts: list of strings; each string is the class, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth.
    """
    imgs_paths, labels, _ = load_image_list(csv_file, data_dir, delimiter)
    filenames = np.asarray(imgs_paths)
    labels = np.asarray(labels, dtype='int32')
    if (not test) and (labels_file is not None):
        categories, _ = helpers.get_lines_in_file(labels_file, read=True)
        texts = np.asarray([categories[label] for label in labels])
    else:
        texts = np.asarray([''] * len(labels))

    return filenames, texts, labels


############################################################
def calc_mean_pixel(filenames):
    sumImg = None
    nImages=0
    print ('Calculating mean image from %d images' % len(filenames))
    for i, filename in enumerate(filenames):
        image = helpers_image.load_and_resize_image(filename, height=0, width=0, mode='RGB')
        if (i%100) == 0:
            print i,
        if (i % 5000) == 0:
            print()
        if sumImg is None:
            h = image.shape[0]
            w = image.shape[1]
            sumImg = np.zeros((h, w, 3), np.float64)  # don't know images sizes yet
        sumImg += image
        nImages+=1

    meanImg = np.round(sumImg / nImages).astype(np.uint8)
    meanPixel = np.mean(meanImg, axis=0)
    meanPixel = np.round(np.mean(meanPixel, axis=0))
    return meanPixel


##################################################################
def find_image_files(data_dir, labels_file, from_file=False, csv_file=None, test=False, start_labels_at_1=False):
    """Build a list of all images files and labels in the data set.

    Args:
      data_dir: string, path to the root directory of images.
      labels_file: string, path to the labels file.

    Returns:
      filenames: list of strings; each string is a path to an image file.
      texts: list of strings; each string is the class, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth.
    """
    if from_file and not (csv_file is None):
        unique_labels = None
        filenames, texts, labels = read_image_files_from_csv(csv_file, data_dir, labels_file, test)
        if start_labels_at_1 and not test:
            labels+=1

    else:
        print('Determining list of input files and labels from %s.' % data_dir)
        unique_labels = [l.strip() for l in tf.gfile.FastGFile(
            labels_file, 'r').readlines()]

        labels = []
        filenames = []
        texts = []

        ## YY:
        # label_index = 1
        if start_labels_at_1:
            # Leave label index 0 empty as a background class.
            label_index = 1
        else:
            label_index = 0

        # Construct the list of JPEG files and labels.
        for text in unique_labels:
            jpeg_file_path = '%s/%s/*' % (data_dir, text)
            matching_files = tf.gfile.Glob(jpeg_file_path)

            labels.extend([label_index] * len(matching_files))
            texts.extend([text] * len(matching_files))
            filenames.extend(matching_files)

            if not label_index % 100:
                print('Finished finding files in %d of %d classes.' % (
                    label_index, len(labels)))
            label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    if not test:
        texts = [texts[i] for i in shuffled_index]
        labels = [labels[i] for i in shuffled_index]

    if unique_labels:
        print('Found %d JPEG files across %d labels inside %s.' %
              (len(filenames), len(unique_labels), data_dir))
    else:
        print('Found %d JPEG files inside %s.' %
              (len(filenames), data_dir))
    return filenames, texts, labels


##################################################################
##################################################################
