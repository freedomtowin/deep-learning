
import tensorflow as tf
import numpy as np
import gnn.gnn_utils as gnn_utils
from gnn.GNN import GNN as GraphNetwork
import gnn.load as ld


##### GPU & stuff config
import os

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#############   DATA LOADING    ##################################################
# function to get a fold
def getFold(fold):
    # load dataset
    train = ld.loadmat("./Data/Mutag/multi" + str(fold))
    train = train['multi' + str(fold)]

    ############ training set #############

    ret_train = gnn_utils.set_load_mutag("train", train)

    ###########validation#####################

    ret_val = gnn_utils.set_load_mutag("validation", train)

    ########### test #####################

    ret_test = gnn_utils.set_load_mutag("test", train)

    return ret_train, ret_val, ret_test


# create the 10-fold in order to train on 10-fold cross validation
tr, val, ts = [], [], []
for fold in range(1, 11):
    a, b, c = getFold(fold)
    tr.append(a)
    val.append(b)
    ts.append(c)


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


# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer

# set parameter
threshold = 0.001
learning_rate = 0.0001
state_dim = 5
max_it = 50
num_epoch = 500
output_dim = 2




testacc = []

for fold in range(0, 10):

    param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
    completeName = param + 'log.txt'
    param = param + "_fold" + str(fold)
    print(param)


    # retrieve input, arcnode, nodegraph and target for training set
    inp = tr[fold][0]
    input_dim = len(inp[0][0])

    arcnode = tr[fold][1]
    labels = tr[fold][4]
    nodegraph = tr[fold][2]

    # retrieve input, arcnode, nodegraph and target for validation set
    inp_val = val[fold][0]
    arcnode_val = val[fold][1]
    labels_val = val[fold][4]
    nodegraph_val = val[fold][2]
    
    inp = inp[0]

    arcnode = arcnode[0]

    nodegraph = nodegraph[0]

    inp_val = inp_val[0]

    arcnode_val = arcnode_val[0]

    nodegraph_val = nodegraph_val[0]

    
    tf.keras.backend.clear_session()
    model = GraphNetwork(input_dim, state_dim, output_dim,                             
                             hidden_state_dim = 15, hidden_output_dim = 10,
                             ArcNode=arcnode,NodeGraph=nodegraph,threshold=threshold)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer,loss_fcn)

    # train GNN, and validate every 2 epochs, (early stopping)
    count = 0
    valid_best = None
    patience = 0
    
    for j in range(0, num_epoch):

        with tf.GradientTape() as tape:
            loss_value = model.train_step(inp.astype(np.float32),labels)

            #early stopping
            if count % 2 == 0:
                out_val = model.predict(inp_val.astype(np.float32), arcnode_val,nodegraph_val)
                loss_value_val = loss_fcn(labels_val,out_val)
                
                if count == 0:
                    valid_best = loss_value_val

                if loss_value_val < valid_best:
                    valid_best = loss_value_val
                    patience = 0
                else:
                    patience += 1

                if patience > 5:
                    print("Early stopping...")
                    break
                    
                count = count + 1

    # retrieve input, arcnode, nodegraph and target for test set
    inp_test = ts[fold][0]
    arcnode_test = ts[fold][1]
    labels_test = ts[fold][4]
    nodegraph_test = ts[fold][2]
    
    inp_test = inp_test[0]

    arcnode_test = arcnode_test[0]

    nodegraph_test = nodegraph_test[0]

    
    print('Accuracy on test set fold ', fold, ' :')

    # evaluate on the test set fold
    
    out_test = model.predict(inp_test, arcnode_test, nodegraph_test)
    metric_value_test = metric(labels_test,out_test)
    testacc.append(metric_value_test.numpy())
    print(metric_value_test.numpy())
#     with open(os.path.join('tmp/', completeName), "a") as file:
#         file.write('Accuracy on test set fold ' + str(fold) + ' :')
#         file.write(str(evel) + '\n')
#         file.write('\n')
#         file.close()

# mean accuracy on the 10-fold
mean_acc = np.mean(np.asarray(testacc))
print('Mean accuracy from all folds:', mean_acc)