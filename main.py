import numpy as np
import pickle
import tensorflow as tf
import datetime

from keras import datasets
from keras import backend as K
from keras.models import load_model

from nrf_corpus import NRFCorpus, NRFElement
from nrf_update import NRFUpdater
from nrf_fetcher import NRFFetcher
from nrf_sampling import NRFSampler
from nrf_mutator import NRFMutator
from nrf_coverage import NRFCoverage
from nrf_metadata import NRFMetadata

from nrf_mutator import L2

from nrf_iterative_fuzzer import NRFFuzzer

tf.flags.DEFINE_string(
    "dataset", None, "Target dataset"
)

tf.flags.DEFINE_integer(
    "iterations", None, "The number of fuzzing iterations, if None, run the fuzzer with given running time"
)
tf.flags.DEFINE_integer(
    "timeout", 28800, "Running time for fuzzer, the fuzzer ends after the given timeout"
)
tf.flags.DEFINE_integer(
    "batchsize", 1, "Batchsize for each fuzzing iteration"
)
tf.flags.DEFINE_string(
    "metric", 'gradfuzz', "Coverage metric to use"
)
tf.flags.DEFINE_string(
    "coverage_mode", 'labeled', "Coverage and fuzzing process type (shared and labeled is available)"
)
tf.flags.DEFINE_string(
    "scope", 'logit', "Scope of layers for coverage (logit, penultimate, and all is available)"
)
tf.flags.DEFINE_boolean(
    "include_loss", True, "Whether the loss will be included in the coverage"
)
tf.flags.DEFINE_boolean(
    "flex_threshold", True, "Whether distance threshold can be flexible"
)
tf.flags.DEFINE_integer(
    "clss", 1, "Mutation class of DeepHunter"
)
tf.flags.DEFINE_boolean(
    "random_seed_corpus", False, "Whether to choose a random seed corpus."
)
FLAGS = tf.flags.FLAGS

checkpoints = {
    'mnist': 'ckpt/lenet5',
    'cifar10': 'ckpt/vgg19',
    'gtsrb': 'ckpt/micronnet'
    }

# 데이터셋 가져오기
def loadData(dataset):
    if dataset == 'mnist':
        numLabels = 10
        targetModel = 'lenet5'

        #from tensorflow.examples.tutorials.mnist import input_data

        #mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
        #train = mnist.train.next_batch(60000)

        #trainImage = np.reshape(np.round(train[0] * 255), (-1, 28, 28, 1)) # np.reshape(train[0] - 0.5, (-1, 28, 28, 1))
        #trainLabel = train[1]

        #test = mnist.test.next_batch(10000)

        #testImage = np.reshape(np.round(test[0] * 255), (-1, 28, 28, 1)) # np.reshape(test[0] - 0.5, (-1, 28, 28, 1))
        #testLabel = test[1]

        def prepMNIST(data):
            temp = np.copy(data)
            temp = np.reshape((temp / 255) - 0.5, (-1, 28, 28, 1))
            return temp

        # Activate when loading dataset
        with open('data/mnist/test_1000.data', 'rb') as rfile:
            test = pickle.load(rfile)
        testImage = test[0]
        testLabel = test[1]

        filterVector = np.ones((28, 28))
        filterVector = np.reshape(filterVector, (28, 28, 1))

        preprocess = prepMNIST

        print('Brought MNIST dataset.')

    elif dataset == 'cifar10':
        numLabels = 10
        targetModel = 'vgg19'

        #(trainImage, trainLabel), (testImage, testLabel) = datasets.cifar10.load_data()
        #img_rows, img_cols, img_ch = trainImage.shape[1:]

        #if K.image_data_format() == 'channels_first':
        #    trainImage = trainImage.reshape(trainImage.shape[0], 1, img_rows, img_cols)
        #    testImage = testImage.reshape(testImage.shape[0], 1, img_rows, img_cols)
        #    input_shape = (1, img_rows, img_cols)
        #else:
        #    trainImage = trainImage.reshape(trainImage.shape[0], img_rows, img_cols, img_ch)
        #    testImage = testImage.reshape(testImage.shape[0], img_rows, img_cols, img_ch)
        #    input_shape = (img_rows, img_cols, 1)

        #trainImage = trainImage.astype('float32')
        #testImage = testImage.astype('float32')
        ##trainImage /= 255
        ##trainImage -= 0.5
        ##testImage /= 255
        ##testImage -= 0.5

        #testLabel = np.reshape(testLabel, (10000,))

        def prepCIFAR10(data):
            temp = np.copy(data)
            temp = np.interp(temp, (0, 255), (-0.5, +0.5))
            return temp

        #Activate when loading dataset
        with open('data/cifar10/test_1000.data', 'rb') as rfile:
            test = pickle.load(rfile)
        testImage = test[0]
        testLabel = test[1]

        filterVector = np.ones((32, 32, 3))

        preprocess = prepCIFAR10

        print('Brought CIFAR-10 dataset.')

    elif dataset == 'gtsrb':
        numLabels = 43
        targetModel = 'micronnet'

        #import gtsrb_pre_data
        #from keras.preprocessing.image import ImageDataGenerator

        #x_train, y_train,x_val, y_val, x_test, y_test = gtsrb_pre_data.pre_data()


        #img_rows, img_cols, img_ch = x_train.shape[1:]

        ##y_train = keras.utils.to_categorical(y_train, numLabels)
        ##y_val = keras.utils.to_categorical(y_val, numLabels)
        ##y_test = keras.utils.to_categorical(y_test, numLabels)

        #train_datagen = ImageDataGenerator(
        #        rotation_range=40,
        #        horizontal_flip=False,
        #        width_shift_range=0.2,
        #        height_shift_range=0.2,
        ##        brightness_range=[0.8,1.0],
        #        shear_range=0.2,
        #        zoom_range=0.2,
        #        fill_mode='nearest',
        #        )
        #validation_datagen = ImageDataGenerator()
        #test_datagen = ImageDataGenerator()

        #aug_data=train_datagen.flow(x_train,y_train,batch_size=50)
        #val_data=validation_datagen.flow(x_val,y_val,batch_size=50)

        #testImage = x_test
        #testLabel = y_test
        ## print(y_test.shape)

        def prepGTSRB(data):
            temp = np.copy(data)
            temp = np.interp(temp, (0, 255), (-0.5, +0.5))
            return temp

        #Activate when loading dataset
        with open('data/gtsrb/test_5000.data', 'rb') as rfile:
            test = pickle.load(rfile)
        testImage = test[0]
        testLabel = test[1]

        filterVector = np.ones((48, 48, 3))

        preprocess = prepGTSRB

        print('Brought GTSRB dataset.')

    else:
        pass

    return testImage, testLabel, numLabels, filterVector, targetModel, preprocess

def loadModel(dataset):
    # 체크포인트 가져오기
    model = load_model('%s/model.h5' % checkpoints[dataset])
    print('Load checkpoint complete.')
    return model

# 플래그 설정
dataset = FLAGS.dataset
iterations = FLAGS.iterations
timeout = FLAGS.timeout
batchsize = FLAGS.batchsize
#flexThreshold = FLAGS.flex_threshold
clss = FLAGS.clss

model = loadModel(dataset)
testImage, testLabel, numLabels, filterVector, targetModel, preprocess = loadData(dataset)

#profiler = Profiler(sess, tensor_map)
#profile = profiler.profile([trainImage, trainLabel])
#profiler.dump('profile/lenet/train.pickle')

with open('%s/profile.pickle' % checkpoints[dataset], 'rb') as rfile:
    profile = pickle.load(rfile)

print('Profiling complete.')

flags = [['gradfuzz-tem', 'shared', 'logit', True, None, 1000, True, 0, 'uniform', 1],
            #['tensorfuzz', 'labeled', 'logit', True, 7200, 500000, True, 0, 'uniform', 1],
            ]
    
testStartTime = datetime.datetime.now()
for flag in flags:

    metric = flag[0]
    mode = flag[1]
    scope = flag[2]
    includeLoss = flag[3]
    timeout = flag[4]
    iterations = flag[5]
    flexThreshold = flag[6]
    clss = flag[7]
    sampletype = flag[8]
    batchsize = flag[9]
    mutation = 'deephunter'

    for execution in range(1):

        import random
        testTemp = [[testImage[i], testLabel[i]] for i in range(len(testImage))]
        random.shuffle(testTemp)
        images = np.array([testTemp[i][0] for i in range(len(testTemp))])
        labels = np.array([testTemp[i][1] for i in range(len(testTemp))])

        fuzzer = NRFFuzzer(execution, targetModel, metric, mode, scope, includeLoss, flexThreshold, mutation, model, preprocess, profile, filterVector, batchsize, numLabels, sampletype, clss=clss)
        # 이니셜시드 돌리기
        initialStartTime = datetime.datetime.now()
        #fuzzer.initialTest(images, labels)
        fuzzer.initialSeed(images, labels)
        initialEndTime = datetime.datetime.now()
        print('Test initial seed complete.')
        initialTime = initialEndTime - initialStartTime
        print('Initial test time : ', initialTime)
        print('Created fuzzer.')

        # 퍼징돌리기
        fuzzingStartTime = datetime.datetime.now()
        fuzzer.doFuzzing(iterations=iterations, timeout=timeout)
        fuzzingEndTime = datetime.datetime.now()
        print('Fuzzing complete.')
        fuzzingTime = fuzzingEndTime - fuzzingStartTime
        print('Fuzzing test time : ', fuzzingTime)
    
        #print(fuzzer.updater.mindist)
        #print(fuzzer.updater.maxdist)
        fuzzer.saveResult(initialTime, fuzzingTime, iterations)
        print('Saved fuzzing result.')
        print('Initial test time : ', initialTime)
        print('Fuzzing test time : ', fuzzingTime)
        print('')

        del fuzzer#, image, label, testTemp
    
testEndTime = datetime.datetime.now()
testTime = testEndTime - testStartTime
print('Total test time : ', testTime)
