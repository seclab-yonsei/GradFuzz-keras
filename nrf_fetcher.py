import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects
from sklearn.preprocessing import OneHotEncoder


class NRFFetcher:
    def act_fetch_function(self, input_batches, label_batches, layer):
        if len(input_batches) == 0:
            return None, None, None, None

        input_batches = self.preprocess(input_batches)

        if layer == 'logit':
            layerIndex = -1
        elif layer == 'penultimate':
            layerIndex = -2
        else:
            layerIndex = -1

        label_batches = np.expand_dims(label_batches, axis=1)
        #print(label_batches)
        label_batches = to_categorical(label_batches, self.numLabels)
        #print(label_batches)

        initialLen = len(input_batches)
        if initialLen < self.batchsize:
            for i in range(initialLen, self.batchsize):
                input_batches = np.append(input_batches, [input_batches[-1]], axis=0)
                label_batches = np.append(label_batches, [label_batches[-1]], axis=0)

        layer_outputs = self.functor1([input_batches, 0]) # activation
        
        activation = [layer_outputs[layerIndex]]
        if layer == 'all':
            activation = layer_outputs
        metadata = layer_outputs[-1]
        prediction = np.argmax(layer_outputs[-1], axis=1)
        #print(activation)

        loss = self.functor2([input_batches, label_batches, 0])

        activation = [act[:initialLen] for act in activation]
        metadata = metadata[:initialLen]
        prediction = prediction[:initialLen]
        loss = loss[:initialLen]

        # Return the prediction outputs
        return activation, metadata, prediction, loss

    # softmax 적용이전 버전
    def grad_fetch_function(self, input_batches, label_batches, layer):
        if len(input_batches) == 0:
            return None, None, None, None

        input_batches = self.preprocess(input_batches)

        initialLen = len(input_batches)
        if initialLen < self.batchsize:
            for i in range(initialLen, self.batchsize):
                input_batches = np.append(input_batches, [input_batches[-1]], axis=0)
                label_batches = np.append(label_batches, [label_batches[-1]], axis=0)

        label_batches = np.expand_dims(label_batches, axis=1)
        #print(label_batches)
        label_batches = to_categorical(label_batches, self.numLabels)
        #print(label_batches)
   

        metadata = self.functor1([input_batches, 0])[0] # activation
        prediction = np.argmax(metadata, axis=1)
        #print(metadata)

        loss = self.functor3([input_batches, label_batches, 0])
        #print(loss)

        gradients = self.functor2([input_batches, label_batches, 0])
        #print(gradients)

        gradients = gradients[:initialLen]
        metadata = metadata[:initialLen]
        prediction = prediction[:initialLen]
        loss = loss[:initialLen]

        # Return the prediction outputs
        return gradients, metadata, prediction, loss
    
    # softamx 적용 버전
    def grad_fetch_function_tem(self, input_batches, label_batches, layer):
        if len(input_batches) == 0:
            return None, None, None, None

        input_batches = self.preprocess(input_batches)

        initialLen = len(input_batches)
        if initialLen < self.batchsize:
            for i in range(initialLen, self.batchsize):
                input_batches = np.append(input_batches, [input_batches[-1]], axis=0)
                label_batches = np.append(label_batches, [label_batches[-1]], axis=0)


        label_batches = np.expand_dims(label_batches, axis=1)
        #print(label_batches)
        label_batches = to_categorical(label_batches, self.numLabels)
        #print(label_batches)
        
        layer_outs_past = self.functor1([input_batches])
        metadata = layer_outs_past[-1]
        # print(metadata)
        prediction = np.argmax(metadata, axis=1)
        # print(prediction)
        # print(metadata)
        # print(layer_outs_past)
        # layer_outs_past = [func([input_batches]) for func in self.functor1]
        # softmax_tem = self.functor2([layer_outs_past[-2]])
        
        
        gradients = self.functor3([layer_outs_past[-2], label_batches])
        # print(np.sum(gradients))

        # metadata = self.functor1([input_batches, 0])[0] # activation
        # prediction = np.argmax(metadata, axis=1)
        #print(metadata)

        loss = self.functor4([input_batches, label_batches, 0])
        # #print(loss)

        # gradients = self.functor2([input_batches, label_batches, 0])
        #print(gradients)

        gradients = gradients[:initialLen]
        metadata = metadata[:initialLen]
        prediction = prediction[:initialLen]
        loss = loss[:initialLen]

        # Return the prediction outputs
        return gradients, metadata, prediction, loss

    def build_fetch_function(self, gradient=True, temperature=False, layer='logit'):
        def func(input_batches, label_batches):
            """The fetch function."""
            if gradient:
                if temperature: 
                    return self.grad_fetch_function_tem(
                        input_batches,
                        label_batches,
                        layer
                )
                else:
                    return self.grad_fetch_function(
                        input_batches,
                        label_batches,
                        layer
                )       
            else:
                return self.act_fetch_function(
                    input_batches,
                    label_batches,
                    layer
                )
        return func

    def __init__(self, model, preprocess, metric, layer='logit', batchsize=1):
        self.model = model
        input = model.input   
        self.preprocess = preprocess
        self.numLabels = model.layers[-1].output_shape[1]
        self.batchsize = batchsize
              
    
        if 'tem' in metric and 'gradfuzz' in metric:
            self.model = model
            input = model.input

            def sample(a):
              a = a*(1/4) # T = 4
              exp_a = K.exp(a)
              p_sum = K.sum(exp_a)
              return exp_a/p_sum

            get_custom_objects().update({'sample': Activation(sample)})

            # lent
            # 추출 모델 생성 및 기존 모델의 w, bias를 추출 모델의 w, bias 설정
            model_ext = Sequential()
            # model_ext.add(Dense(84, input_shape=(256,), name="dense_0"))
            
            model_ext.add(Dense(model.layers[-1].output_shape[1], input_shape=(model.layers[-2].output_shape[1],), name="dense_1", activation=sample))
            adam = Adam(lr=5e-4)
            model_ext.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

            # model_ext.get_layer("dense_0").set_weights([model.layers[-2].get_weights()[0], model.layers[-2].get_weights()[1]])
            model_ext.get_layer("dense_1").set_weights([model.layers[-1].get_weights()[0], model.layers[-1].get_weights()[1]])
            model_ext.save('model_lenet_extract.h5')
            inp = model_ext.input

            # 원래 모델에서 logit레이어 이전 아웃풋 값 추출
            # outputs_past = [layer.output for layer in model.layers]
            # functor1 = [K.function([input],[out]) for out in outputs_past]
            # self.functor1 = K.function([input], outputs_past)
            self.functor1 = K.function([input], [layer.output for layer in model.layers])
            # print(model.layers)

            self.functor2 = K.function([inp],[layer.output for layer in model_ext.layers])
            
            
            label_tensor = tf.placeholder(tf.float32, shape=(None, 1))
            loss = keras.losses.categorical_crossentropy(label_tensor, model_ext.layers[-1].output)
            losses = tf.split(loss, batchsize)   
            gradient_tensor = list()

            for index in range(batchsize):
              gradient_tensor.extend(K.gradients(losses[index], model_ext.trainable_weights[-2]))
            self.functor3 = K.function([inp, label_tensor] + [K.learning_phase()], gradient_tensor)


            loss = keras.losses.categorical_crossentropy(label_tensor, model.layers[-1].output)
            losses = tf.split(loss, batchsize)
            self.functor4 = K.function([input, label_tensor] + [K.learning_phase()], losses)
            
            
            self.fetchFunction = self.build_fetch_function(temperature=True, layer='logit')
        
        elif 'gradfuzz' in metric:
            self.model = model
            input = model.input
            label_tensor = tf.placeholder(tf.float32, shape=(None, 1))
            loss = keras.losses.categorical_crossentropy(label_tensor, model.layers[-1].output)
            losses = tf.split(loss, batchsize)
            
            self.functor1 = K.function([input] + [K.learning_phase()], [model.layers[-1].output])
    
            # loss가 뭔지 궁금할 때
            gradient_tensor = list()
            for index in range(batchsize):
                gradient_tensor.extend(K.gradients(losses[index], model.trainable_weights[-2]))
            self.functor2 = K.function([input, label_tensor] + [K.learning_phase()], gradient_tensor)
            self.functor3 = K.function([input, label_tensor] + [K.learning_phase()], losses)
            
            self.fetchFunction = self.build_fetch_function(gradient=True, layer='logit')

        else:
            self.model = model
            input = model.input
            label_tensor = tf.placeholder(tf.float32, shape=(None, 1))
            loss = keras.losses.categorical_crossentropy(label_tensor, model.layers[-1].output)
            losses = tf.split(loss, batchsize)

            self.functor1 = K.function([input] + [K.learning_phase()], [layer.output for layer in model.layers])
            self.functor2 = K.function([input, label_tensor] + [K.learning_phase()], losses)

            self.fetchFunction = self.build_fetch_function(gradient=False, layer=layer)
