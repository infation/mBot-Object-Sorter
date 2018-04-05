import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from random import shuffle
import os
from time import sleep
import serial
import tensorflow as tf
from picamera import PiCamera

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
st = lambda aug: iaa.Sometimes(0.3, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

seq2 = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        st(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
        st(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
        st(iaa.GaussianBlur((0, 3.0))), # blur images with a sigma between 0 and 3.0
        st(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))), # sharpen images
        st(iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))), # emboss images
        st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)), # add gaussian noise to images
        st(iaa.Dropout((0.0, 0.1), per_channel=0.5)), # randomly remove up to 10% of the pixels
        st(iaa.Add((-20, 20), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
        st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
        st(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)) # apply elastic transformations with random strengths
    ],
    random_order=True # do all of the above in random order
)

TEST_DIR = '/home/pi/Desktop/Robotics-final/test'
IMG_SIZE = 32
LR = 0.0008
training_data = []

def label_in_dir(directory):
	#print(directory)
	if directory ==   'bluecube':return       [1,0,0,0,0,0,0,0,0,0,0,0]
	elif directory == 'bluepyr': return       [0,1,0,0,0,0,0,0,0,0,0,0] #ALB 
	elif directory == 'bluesphere': return    [0,0,1,0,0,0,0,0,0,0,0,0] #BET
	elif directory == 'greencube': return     [0,0,0,1,0,0,0,0,0,0,0,0] #BET
	elif directory == 'greenpyr': return      [0,0,0,0,1,0,0,0,0,0,0,0] #BET
	elif directory == 'greensphere': return   [0,0,0,0,0,1,0,0,0,0,0,0] #BET
	elif directory == 'redcube': return       [0,0,0,0,0,0,1,0,0,0,0,0] #BET
	elif directory == 'redpyr': return        [0,0,0,0,0,0,0,1,0,0,0,0] #BET
	elif directory == 'redsphere': return     [0,0,0,0,0,0,0,0,1,0,0,0] #BET
	elif directory == 'yellowcube': return    [0,0,0,0,0,0,0,0,0,1,0,0] #BET
	elif directory == 'yellowpyr': return     [0,0,0,0,0,0,0,0,0,0,1,0] #BET
	elif directory == 'yellowsphere': return  [0,0,0,0,0,0,0,0,0,0,0,1] #BET
#################LABELING THE TRAINING DATA##################################
def label_class(label, directory):
        for img in tqdm(os.listdir(directory)):
                images = []
                path = os.path.join(directory, img)
                image = cv2.imread(path)
                images.append(image)
                img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

                # convert the YUV image back to RGB format
                img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                images.append(img_output)
                img = cv2.resize(img_output, (IMG_SIZE, IMG_SIZE))
                #Produce 10 more augmented images for each train image
                for i in range(10):
                        image_aug = seq.augment_images(images)
                        img_aug1 = cv2.resize(image_aug[0], (IMG_SIZE, IMG_SIZE))
                        img_aug2 = cv2.resize(image_aug[1], (IMG_SIZE, IMG_SIZE))
                        training_data.append([np.array(img_aug1).flatten(),np.array(label)])
                        training_data.append([np.array(img_aug2).flatten(),np.array(label)])
                
                training_data.append([np.array(img).flatten(), np.array(label)])


def create_train_data():
        for directory in tqdm(os.listdir(TRAIN_DIRS)):
                print(directory)
                path = os.path.join(TRAIN_DIRS, directory)
                label = label_in_dir(directory)
                label_class(label, path)
        shuffle(training_data)
        np.save('all_train_data_bigaugm.npy', training_data)
        return training_data

#######################PROCESS EACH IMAGE THE RASPBERRY CAMERA TAKES###################
def process_test_data():
	testing_data = []
	for img in os.listdir(TEST_DIR):
		path = os.path.join(TEST_DIR, img)
		image = cv2.imread(path)
                #Apply histogram equalization 
		img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
		img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
		img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
		img_num = img
		img = cv2.resize(img_output, (IMG_SIZE, IMG_SIZE))
		testing_data.append([np.array(img).flatten(), img_num])

	np.save('test_data.npy', testing_data)
	return testing_data

###########################NEURAL NETWORK TRAINING############################################

#USED FOR TRAINING THE NEURAL NET AND DIVIDING INTO TRAIN AND TEST
#train_data = create_train_data()
#train_data = np.load('all_train_data_bigaugm.npy')

#train = train_data[:-200]
#test = train_data[-200:]

#train_x = np.array([i[0] for i in train], dtype=np.float32)
#train_y = np.array([i[1] for i in train])

#test_x = np.array([i[0] for i in test], dtype=np.float32)
#test_y = np.array([i[1] for i in test])

n_classes = 12
batch_size = 50

x = tf.placeholder(tf.float32, shape = [None, 3*IMG_SIZE*IMG_SIZE])
y = tf.placeholder(tf.float32)

keep_rate = 0.9
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
	#                        size of window         movement of window
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def get_next_batch(i):
	start = i*batch_size
	return train_x[start:start+batch_size], train_y[start:start+batch_size]

def convolutional_neural_network(x):
   
	weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),
			   'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
			   'W_conv3':tf.Variable(tf.random_normal([5,5,64,128])),
			   'W_fc':tf.Variable(tf.random_normal([4*4*128,1024])),
			   'out':tf.Variable(tf.random_normal([1024, n_classes]))}

	biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
			   'b_conv2':tf.Variable(tf.random_normal([64])),
			   'b_conv3':tf.Variable(tf.random_normal([128])),
			   'b_fc':tf.Variable(tf.random_normal([1024])),
			   'out':tf.Variable(tf.random_normal([n_classes]))}


	x = tf.reshape(x, shape = [-1, IMG_SIZE, IMG_SIZE, 3])
	#print (x)
	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
	conv1 = maxpool2d(conv1)
	
	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
	conv2 = maxpool2d(conv2)

	conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
	conv3 = maxpool2d(conv3)

	fc = tf.reshape(conv3, shape = [-1, 4*4*128])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights['out'])+biases['out']

	return output

#PROCESS IMAGE THAT THE PI CAMERA TOOK, 
#RUN TRHOUGH THE NET, CLASSIFY AND SEND BACK TO THE ARDUINO
def classify_object():
        y = convolutional_neural_network(x)
        prediction = tf.argmax(y, 1)
        init = tf.global_variables_initializer()
        con = serial.Serial('/dev/ttyUSB0', 9600)
        sleep(1)
        con.setDTR(level=0)
        sleep(1)
        camera = PiCamera()
        camera.vflip = True
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess, '/home/pi/Desktop/Robotics-final/3_layer_big_augment.ckpt')
                while True:
                        while con.inWaiting():
                                print(con.read())
                        #sleep(0.2)
                        msg = con.read()
                        if msg == '0':
                                camera.start_preview()
                                sleep(3)
                                camera.capture('/home/pi/Desktop/Robotics-final/test/foo.jpg')
                                camera.stop_preview()
                                sleep(0.5)
                                test_data = process_test_data()
                                #print(test_data)
                                for data in test_data:
                                        img_data = np.array(data[0], dtype = 'float32').flatten()
                                        data = img_data.reshape(1, 3*IMG_SIZE*IMG_SIZE)
                                        
                                        pred = prediction.eval(feed_dict={x:data})[0]
                                        print(pred)
                                        if pred == 0:
                                                con.write('0')  #bluecube
                                        elif pred == 1:
                                                con.write('1')  #bluepyr
                                        elif pred == 2:
                                                con.write('2')  #bluesphere
                                        elif pred == 3:
                                                con.write('3')  #greencube
                                        elif pred == 4:
                                                con.write('4')  #greenpyr
                                        elif pred == 5:
                                                con.write('5')  #greensphere
                                        elif pred == 6:
                                                con.write('6')  #redcube
                                        elif pred == 7:
                                                con.write('7')  #redpyr
                                        elif pred == 8:
                                                con.write('8')  #redsphere
                                        elif pred == 9:
                                                con.write('9')  #yellowcube
                                        elif pred == 10:
                                                con.write('a')  #yellowpyr
                                        elif pred == 11:
                                                con.write('b')  #yellowsphere

# USED FOR TRAINING THE MODEL
def train_neural_network(x):
        prediction = convolutional_neural_network(x)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
        optimizer = tf.train.AdamOptimizer(LR).minimize(cost)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        hm_epochs = 20

        with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess, 'C:/Users/Stan/Desktop/Robotics-final/3_layer_augment.ckpt')
                print("Model restored.")
                for epoch in range(hm_epochs):
                        shuffle(train)
                        #shuffle(test)
                        epoch_loss = 0
                        for i in tqdm(range(int(len(train)/batch_size))):

                                epoch_x, epoch_y = get_next_batch(i)
                                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                                epoch_loss += c

                        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
                        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                        a = accuracy.eval({x:test_x, y:test_y})
                        print('Accuracy:', a)
        
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

                print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))
                save_path = saver.save(sess, 'C:/Users/Stan/Desktop/Robotics-final/3_layer_big_augment.ckpt')
                print("Model saved in file")


classify_object()
