from scipy import misc as imageIo
import numpy as np
import random
import os

class InputCreation:
    def _init_(self):
        self.train_pairs = []
        self.train_labels = []
        self.test_pairs = []
        self.test_labels = []
        self.total_images = 0

    def train_input(self):
        return self.train_pairs

    def train_output(self):
        return self.train_labels

    def test_input(self):
        return self.test_pairs

    def test_output(self):
        return self.test_labels

    def read_image_files(self, feature_file, data_path):
        final = {}
        # Grab all the file names for all the images.
        img_file_names = os.listdir(data_path)
        # Extracting all the features from all the images.
        for file_name in img_file_names:

            img = imageIo.imread(data_path + '/' + file_name, flatten=True)
            img = imageIo.imresize(img, (28, 28))
            img = img / 255
            img = img.flatten()
            img[:] = 1 - img[:]
            img = np.asarray(img)
            writer_id = file_name[0:4]

            # Add the image vector to the dictionary for the specific writer.
            if writer_id not in final:
                image_feature_list = [img]
            else:
                image_feature_list = final.get(writer_id)
                image_feature_list.append(img)

            # Update the dictionary for the new image that is read for the new writer.
            final.update({writer_id: image_feature_list})
        np.save(feature_file, final)

    def generate_input_output_dataset(self, feature_file, numPairs):
        # lists to store inputs and outputs
        inputs = []
        outputs = []

        # read the descriptor file in dictionary
        descriptorDict = np.load(feature_file).item()

        # fetch all the keys in a list
        keysList = [*descriptorDict]

        # iterating all the keys in the dictionary
        # forming same writer/different writer pairs
        for k, v in descriptorDict.items():
            # number of images for a writer
            numImages = len(v)
            # if > 1 then form same writer pairs
            if numImages > 1:
                # random sampling of images to make pairs
                # max number of pairs = 5
                l1 = random.sample(range(numImages), min(numImages, numPairs))
                l2 = random.sample(range(numImages), min(numImages, numPairs))
                random.shuffle(l2)

                # forming pairs
                for (i1, i2) in zip(l1, l2):
                    if i1 != i2:
                        #inputs.append(np.concatenate([v[i1].flatten(),v[i2].flatten()]))
                        inputs.append([v[i1], v[i2]])
                        outputs.append([1,0])

            # fetching 5 writers from the keysList
            negativeSamples = [keysList[i1] for i1 in random.sample(range(len(keysList)), numPairs)]

            # iterating to form different writer pairs
            for key in negativeSamples:
                if key != k:
                    # getting descriptors list for different writer
                    descList = descriptorDict.get(key)

                    # randomly sampling an image for each writer
                    i1 = random.randint(0, len(descList)-1)
                    i2 = random.randint(0, len(v)-1)

                    # forming pairs
                    inputs.append([descList[i1], v[i2]])
                    outputs.append([0, 1])

        print('same writer:', outputs.count([1,0]), 'different writer:',
            outputs.count([0,1]))
        inputs = np.asarray(inputs)
        outputs = np.asarray(outputs)
        p = np.arange(inputs.shape[0])
        np.random.shuffle(p)
        inputs = inputs[p]
        outputs = outputs[p]
        marker = int(len(inputs) * 0.8)
        self.train_pairs = inputs[:marker]
        self.train_labels = outputs[:marker]
        self.test_pairs = inputs[marker:]
        self.test_labels = outputs[marker:]

