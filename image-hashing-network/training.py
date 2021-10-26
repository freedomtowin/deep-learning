import numpy as np
import tensorflow as tf
import random
import cv2 
from tensorflow.python.framework import ops

# @tf.function()
# def tf_weight_mse(x1, x2, y):
#     f1 = tf.keras.layers.Flatten()
#     x1 = f1(x1)
#     x1 = f1(x2)
    
#     loss = tf.reduce_mean((x2-x1)*(x2-x1)*y)
#     return loss

@tf.function()
def tf_weight_mse(x1, x2, y):
    loss = tf.reduce_mean((x2-x1)*(x2-x1)*y)
    return loss

class EmbeddingLayer(tf.keras.Model):
    def __init__(self):
        super(EmbeddingLayer,self).__init__(name='')
    
        self.c1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")
        self.c2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")
        self.c3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")
        self.m1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.c4 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, activation="relu")
        self.f1 = tf.keras.layers.Flatten()
#         self.c4 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, activation="relu")
    def call(self, data):

        c1 = self.c1(data)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        m1 = self.m1(c3)
        c4 = self.c4(m1)
        f1 = self.f1(m1)

        return f1
    
    

    
class ImageMatcher(tf.keras.Model):
    def __init__(self,IMG_SHAPE,img_idx_to_base_idxs):
        super(ImageMatcher,self).__init__(name='')
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.loss_fcn=tf_weight_mse#tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.IMG_SHAPE = IMG_SHAPE
        self.img_idx_to_base_idxs = img_idx_to_base_idxs
        self.base_imgs = list(self.img_idx_to_base_idxs.keys())
        self.create_model()
        
    def create_model(self):
        input_1 = tf.keras.layers.Input(shape=(self.IMG_SHAPE, self.IMG_SHAPE, 3))
        input_2 = tf.keras.layers.Input(shape=(self.IMG_SHAPE, self.IMG_SHAPE, 3))
        self.embeddings = EmbeddingLayer()
        output_1 = self.embeddings(input_1)
        output_2 = self.embeddings(input_2)
        inputs = [input_1,input_2]
        outputs = [output_1,output_2]
        self.model = tf.keras.Model(inputs,outputs)
        
    def get_anchor_pair(self):
        
        if np.random.uniform()>0.3:
            random_class = random.choice(self.base_imgs)

            examples_for_class = list(self.img_idx_to_base_idxs[random_class])
    #         print(examples_for_class,random_class)
            anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            if positive_idx==anchor_idx:
                positive_idx = random.choice(examples_for_class)
#             print(anchor_idx,positive_idx)
    #         x[0] = cv2.imread(anchor_idx)
    #         x[1] = cv2.imread(positive_idx)

            return cv2.imread(anchor_idx),cv2.imread(positive_idx),1.
        else:
            random_class_1 = random.choice(self.base_imgs)
            random_class_2 = random.choice(self.base_imgs)
            if random_class_1==random_class_2:
                random_class_1 = random.choice(self.base_imgs)

            examples_for_class_1 = list(self.img_idx_to_base_idxs[random_class_1])
            examples_for_class_2 = list(self.img_idx_to_base_idxs[random_class_2])
    #         print(examples_for_class,random_class)
            anchor_idx = random.choice(examples_for_class_1)
            negative_idx = random.choice(examples_for_class_2)
            
            return cv2.imread(anchor_idx),cv2.imread(negative_idx),-1.

    def train_step(self):
        batch_size=100
        X1 = np.empty((batch_size, self.IMG_SHAPE, self.IMG_SHAPE, 3), dtype=np.float32)
        X2 = np.empty((batch_size, self.IMG_SHAPE, self.IMG_SHAPE, 3), dtype=np.float32)
        Y = np.empty((batch_size,1), dtype=np.float32)

        for i in range(batch_size):
            img_1,img_2,y = self.get_anchor_pair()
            X1[i] = img_1/255.
            X2[i] = img_2/255.
            Y[i] = y
#         print(Y)
#         if np.random.uniform()>0.3:
#             img_1,img_2,wgt = next(iter(AnchorPositivePairs(num_batchs=1)))
#         else:
#             img_1,img_2,wgt = next(iter(AnchorNegativePairs(num_batchs=1)))

        with tf.GradientTape() as tape:
                    # Run both anchors and positives through model.
            anchors,positives = self.model([X1,X2])
            
            anchor_embeddings = anchors
            positive_embeddings = positives
            
#             similarities = tf.einsum(
#                 "ae,pe->ap", anchor_embeddings, positive_embeddings
#             )

#             # Since we intend to use these as logits we scale them by a temperature.
#             # This value would normally be chosen as a hyper parameter.
#             temperature = 0.2
#             similarities /= temperature
#             loss = self.loss_fcn(Y,similarities)            
#             anchor_embeddings = anchors[0]
#             positive_embeddings = positives[0]

            loss = self.loss_fcn(anchor_embeddings,positive_embeddings, Y)
            print('loss: ',loss.numpy())
            # Calculate gradients and apply via optimizer.
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
