import random
import numpy as np
import time
import tensorflow as tf
import input_creation
import math
import csv
from scipy import misc as imageIo


def mlp(input_,input_dim,output_dim,name="mlp"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))
        return tf.nn.relu(tf.matmul(input_,w))
        

def build_model_mlp(X_,_dropout):
    model = mlpnet(X_,_dropout)
    return model


def mlpnet(image,_dropout):
    l1 = mlp(image,784,128,name='l1')
    l1 = tf.nn.dropout(l1,_dropout)
    l2 = mlp(l1,128,128,name='l2')
    l2 = tf.nn.dropout(l2,_dropout)
    l3 = mlp(l2,128,128,name='l3')
    return l3


def contrastive_loss(y,d):
    tmp= y *tf.square(d)
    batch_size = 128
    #tmp= tf.mul(y,tf.square(d))
    tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
    return tf.reduce_sum(tmp +tmp2)/batch_size/2


def compute_accuracy(prediction,labels):
    return labels[prediction.ravel() < 0.5].mean()
    #return tf.reduce_mean(labels[prediction.ravel() < 0.5])


def next_batch(s,e,inputs,labels):
    input1 = inputs[s:e,0]
    input2 = inputs[s:e,1]
    temp_labels = labels[s:e]
    y = []
    for label in temp_labels:
        y.append(label[0])
    y = np.asarray(y)
    y = np.reshape(y, (y.shape[0], 1))
    return input1,input2,y

def train_model(test_pairs, test_output):
    # Initializing the variables
    init = tf.initialize_all_variables()
    batch_size =128
    global_step = tf.Variable(0,trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase=True)
    # File path where the images are stored.
    DATA_PATH = '../AND_dataset/Dataset[Without-Features]/AND_Images[WithoutFeatures]/'
    FEATURE_FILE = './featureFile'

    # Get the input and output positive + negative pairs.
    input_creator = input_creation.InputCreation()
    input_creator.read_image_files(FEATURE_FILE, DATA_PATH)
    input_creator.generate_input_output_dataset(FEATURE_FILE + '.npy', 14)
    tr_pairs = input_creator.train_input()
    tr_y = input_creator.train_output()
    te_pairs = input_creator.test_input()
    te_y = input_creator.test_output()

    # Create TF variables for the model.
    images_L = tf.placeholder(tf.float32,shape=([None,784]),name='L')
    images_R = tf.placeholder(tf.float32,shape=([None,784]),name='R')
    labels = tf.placeholder(tf.float32,shape=([None,1]),name='gt')
    dropout_f = tf.placeholder("float")

    # Reuse the model parameterr to build a Sianese network.
    with tf.variable_scope("SiameseNetwork") as scope:
        model1= build_model_mlp(images_L,dropout_f)
        scope.reuse_variables()
        model2 = build_model_mlp(images_R,dropout_f)

    distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model1,model2),2),1,keep_dims=True))
    loss = contrastive_loss(labels,distance)
    #contrastice loss
    t_vars = tf.trainable_variables()
    d_vars  = [var for var in t_vars if 'l' in var.name]
    batch = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)

    # The model saver.
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        # Initialise all the variables for the session.
        tf.initialize_all_variables().run()

        # Training cycle
        for epoch in range(200):
            avg_loss = 0.
            avg_acc = 0.
            total_batch = int(tr_pairs.shape[0]/batch_size)
            start_time = time.time()

            # Loop over all batches
            for i in range(total_batch):
                s = i * batch_size
                e = (i+1) *batch_size

                # Fit training using batch data
                input1,input2,y =next_batch(s,e,tr_pairs,tr_y)
                _,loss_value,predict=sess.run([optimizer,loss,distance], feed_dict={images_L:input1,images_R:input2 ,labels:y,dropout_f:0.9})
                feature1 = model1.eval(feed_dict={images_L:input1,dropout_f:0.9})
                feature2 = model2.eval(feed_dict={images_R:input2,dropout_f:0.9})
                tr_acc = compute_accuracy(predict,y)
                if math.isnan(tr_acc) and epoch != 0:
                    print('tr_acc %0.2f' % tr_acc)
                avg_loss += loss_value
                avg_acc += tr_acc*100

            duration = time.time() - start_time
            print('epoch %d  time: %f loss %0.5f acc %0.2f' %(epoch,duration,avg_loss/(total_batch),avg_acc/total_batch))
        save_path = saver.save(sess, "./model.ckpt")
        print("Model saved in path: %s" % save_path)

        # Calculate the training accuracy.
        y = []
        for label in tr_y:
            y.append(label[0])
        y = np.asarray(y)
        y = np.reshape(y, (y.shape[0], 1))
        print(tr_pairs.shape)
        predict = distance.eval(feed_dict={images_L:tr_pairs[:,0],images_R:tr_pairs[:,1],labels:y,dropout_f:1.0})
        tr_acc = compute_accuracy(predict,y)
        print('Accuracy training set %0.2f' % (100 * tr_acc))

        # Test model
        y = []
        for label in te_y:
            y.append(label[0])
        y = np.asarray(y)
        y = np.reshape(y, (y.shape[0], 1))
        predict = distance.eval(
            feed_dict={images_L: te_pairs[:, 0], images_R: te_pairs[:, 1], labels: y, dropout_f: 1.0})
        te_acc = compute_accuracy(predict,y)
        print('Accuracy test set %0.2f' % (100 * te_acc))

        # Test model on the hidden data.
        y = []
        for label in test_output:
            y.append(label[0])
        y = np.asarray(y)
        y = np.reshape(y, (y.shape[0], 1))
        predict = distance.eval(
            feed_dict={images_L: test_pairs[:, 0], images_R: test_pairs[:, 1], labels: y, dropout_f: 1.0})
        te_acc = compute_accuracy(predict, y)
        print('Accuracy hidden test set %0.2f' % (100 * te_acc))


if __name__ == "__main__":
    print("Training the model with random pairs along with 200 epochs. After the training, the model will be tested on the hidden data...")
    # Read the test data shared and process the accuracy.
    test_data_specs = './DLHiddenTest/DLTestOutput.csv'
    MODEL_CHECKPOINT = './model.ckpt'
    test_pairs = []
    test_output = []
    with open(test_data_specs, 'r') as csvfile:
        # Read the csv rows
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:
            if i != 0:
                image_one_name = row[1]
                image_two_name = row[2]
                same_writer_flag = row[4]
                img1 = imageIo.imread('./DLHiddenTest/DLTestData/' + image_one_name, flatten=True)
                img1 = imageIo.imresize(img1, (28, 28))
                img1 = img1 / 255
                img1 = img1.flatten()
                img1[:] = 1 - img1[:]
                img1 = np.asarray(img1)
                img2 = imageIo.imread('./DLTestData/' + image_two_name, flatten=True)
                img2 = imageIo.imresize(img2, (28, 28))
                img2 = img2 / 255
                img2 = img2.flatten()
                img2[:] = 1 - img2[:]
                img2 = np.asarray(img2)

                # Create pair.
                test_pairs.append([img1, img2])
                if same_writer_flag == 1:
                    test_output.append([1,0])
                else:
                    test_output.append([0,1])
            i = i + 1

        # Convert to numpy arrays\.
        test_pairs = np.asarray(test_pairs)
        test_output = np.asarray(test_output)

        # Train the model and test on hidden data.
        train_model(test_pairs, test_output)