
import tensorflow as tf
import numpy as np
import gnn.gnn_utils as utils
from gnn.GNN import GNN as GraphNetwork


# import tensorflow as tf
# import numpy as np
# import utils
# import GNNs as GNN
# import Net_Karate as n
# from scipy.sparse import coo_matrix

##### GPU & stuff config
import os

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


############# training set ################


E, N, labels,  mask_train, mask_test = utils.load_karate()
inp, arcnode, graphnode = utils.from_EN_to_GNN(E, N)



EPSILON = 0.00000001

@tf.function()
def loss_fcn(target,output,mask):
    target = tf.cast(target,tf.float32)
    output = tf.maximum(output, EPSILON, name="Avoiding_explosions")  # to avoid explosions
    xent = -tf.reduce_sum(target * tf.math.log(output), 1)

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    xent *= mask
    lo = tf.reduce_mean(xent)
    return lo

@tf.function()
def metric(output, target,mask):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask

    return tf.reduce_mean(accuracy_all)




# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
threshold = 0.001
learning_rate = 0.0001
state_dim = 2
input_dim = inp.shape[1]
output_dim = labels.shape[1]
max_it = 50
num_epoch = 1000

tf.keras.backend.clear_session()
model = GraphNetwork(input_dim, state_dim, output_dim,                             
                         hidden_state_dim = 15, hidden_output_dim = 10,
                         ArcNode=arcnode,NodeGraph=None,threshold=threshold)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer,loss_fcn)


# initialize GNN
param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
print(param)


for count in range(0, num_epoch):
    loss_value = model.train_step(inp.astype(np.float32),labels, mask=mask_train)
    
    if count % 30 == 0:
        #this runs the loop without training
        out = model.predict(inp.astype(np.float32), arcnode)
        loss_value_val = loss_fcn(labels,out, mask=mask_test)

        print("Epoch ", count)
        print("Training: ", loss_value.numpy())
        print("Validation: ",loss_value_val.numpy())

        count = count + 1