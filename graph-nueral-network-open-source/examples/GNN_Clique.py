import tensorflow as tf
import numpy as np
import gnn.gnn_utils as gnn_utils
from gnn.GNN import GNN as GraphNetwork



##### GPU & stuff config
import os

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


data_path = "./data"

############# training set ################

set_name = "cli_15_7_200"

#inp, arcnode, nodegraph, nodein, labels = Library.set_load_subgraph(data_path, "train")
inp, arcnode, nodegraph, nodein, labels, _ = gnn_utils.set_load_general(data_path, "train", set_name=set_name)
############ test set ####################

#inp_test, arcnode_test, nodegraph_test, nodein_test, labels_test = Library.set_load_subgraph(data_path, "test")
inp_test, arcnode_test, nodegraph_test, nodein_test, labels_test, _ = gnn_utils.set_load_general(data_path, "test", set_name=set_name)

############ validation set #############

#inp_val, arcnode_val, nodegraph_val, nodein_val, labels_val = Library.set_load_subgraph(data_path, "valid")
inp_val, arcnode_val, nodegraph_val, nodein_val, labels_val, _ = gnn_utils.set_load_general(data_path, "validation", set_name=set_name)



EPSILON = 0.00000001

@tf.function()
def loss_fcn(target,output):
    target = tf.cast(target,tf.float32)
    output = tf.maximum(output, EPSILON, name="Avoiding_explosions")  # to avoid explosions
    xent = -tf.reduce_sum(target * tf.math.log(output), 1)
    lo = tf.reduce_mean(xent)
    return lo

@tf.function()
def metric(output, target):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
    metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return metric


inp = inp[0]

arcnode=arcnode[0]

nodegraph=nodegraph[0]

inp_val = inp_val[0]

arcnode_val = arcnode_val[0]

nodegraph_val=nodegraph_val[0]

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
threshold = 0.001
learning_rate = 0.01
state_dim = 5
input_dim = len(inp[0])
output_dim = 2
max_it = 50
num_epoch = 500


# initialize GNN
param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
print(param)


model = GraphNetwork(input_dim, state_dim, output_dim,                             
                         hidden_state_dim = 15, hidden_output_dim = 10,
                         ArcNode=arcnode,NodeGraph=None,threshold=threshold)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer,loss_fcn)



for count in range(0, num_epoch):
    loss_value = model.train_step(inp.astype(np.float32),labels)
    
    if count % 30 == 0:
        #this runs the loop without training
        out_val = model.predict(inp_val.astype(np.float32), arcnode_val)
        loss_value_val = loss_fcn(labels_val,out_val)

        print("Epoch ", count)
        print("Training: ", loss_value.numpy())
        print("Validation: ",loss_value_val.numpy())

        count = count + 1