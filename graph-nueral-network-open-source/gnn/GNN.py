import tensorflow as tf
import numpy as np
import datetime as time

class GNN(tf.keras.Model):
    def __init__(self, input_dim, state_dim, output_dim, hidden_state_dim, hidden_output_dim, ArcNode, NodeGraph=None, threshold=None):
        super(GNN,self).__init__(name='')
        # initialize weight and parameter
        
        self.graph_based = False
        
        if threshold is None:
            self.state_threshold = 0.01
        else:
            self.state_threshold = threshold
            
        self.max_iter = 50

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.state_input = self.input_dim - 1 + state_dim  # removing the id_ dimension

        #### TO BE SET ON A SPECIFIC PROBLEM
        self.state_l1 = hidden_state_dim
        self.state_l2 = self.state_dim

        self.output_l1 = hidden_output_dim
        self.output_l2 = self.output_dim
        
        self.ArcNode = tf.sparse.SparseTensor(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        
        if NodeGraph is not None:
            self.graph_based = True
            self.NodeGraph = tf.sparse.SparseTensor(indices=NodeGraph.indices, values=NodeGraph.values,
                                        dense_shape=NodeGraph.dense_shape)

        self.inp = tf.keras.Input(shape=(input_dim), name="input")
        
        self.state_init = np.zeros((ArcNode.dense_shape[0], state_dim))
        
        self.state_old_init = np.ones((ArcNode.dense_shape[0], state_dim))
        
        
        self.state = tf.Variable(self.state_init, name="state",trainable=False,dtype=tf.float32)

        self.state_old = tf.Variable(self.state_old_init, name="old_state",trainable=False,dtype=tf.float32)
    
        self.layer_1_state = tf.keras.layers.Dense(self.state_l1, activation='tanh',name='state_1')
        self.layer_2_state = tf.keras.layers.Dense(self.state_l2, activation='tanh',name='state_2')
        self.layer_1_out = tf.keras.layers.Dense(self.output_l1, activation='tanh',name='out_1')
        self.layer_2_out = tf.keras.layers.Dense(self.output_l2, activation='softmax',name='out_2')
        
        
        
    def compile(self, optimizer, loss_fn):
        super(GNN, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn      
       
    def convergence(self, a, state, old_state, k):

        # assign current state to old state
        old_state = state

        # grub states of neighboring node
        gat = gather_dense_gradient(old_state, tf.cast(a[:, 1], tf.int32))

        
        # slice to consider only label of the node and that of it's neighbor
        # sl = tf.slice(a, [0, 1], [tf.shape(a)[0], tf.shape(a)[1] - 1])
        # equivalent code
        sl = a[:, 2:]


        # concat with retrieved state
        inp = tf.cast(tf.concat([sl, gat], axis=1),tf.float32)

        # evaluate next state and multiply by the arch-node conversion matrix to obtain per-node states
        layer_1_state = self.layer_1_state(inp)
        layer_2_state = self.layer_2_state(layer_1_state)

        state = tf.cast(tf.sparse.sparse_dense_matmul(self.ArcNode, layer_2_state),tf.float32)

        # update the iteration counter
        k = k + 1
        return a, state, old_state, k

    def condition(self, a, state, old_state, k):
        # evaluate condition on the convergence of the state

        # evaluate distance by state(t) and state(t-1)
        outDistance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, old_state)), 1) + 0.00000000001)
        # vector showing item converged or not (given a certain threshold)
        checkDistanceVec = tf.greater(outDistance, self.state_threshold)

        c1 = tf.reduce_any(checkDistanceVec)
        c2 = tf.less(k, self.max_iter)

        return tf.logical_and(c1, c2)
    
    def call(self, comp_inp):

        k = tf.constant(0)

        self.state.assign(self.state_init)
        self.state_old.assign(self.state_old_init)
        
        res, st, old_st, num = tf.while_loop(self.condition, self.convergence,
                                                 [comp_inp, self.state, self.state_old, k])
        
            
        # grub states of neighboring node
        gat = gather_dense_gradient(st, tf.cast(comp_inp[:, 1], tf.int32))

        
        # slice to consider only label of the node and that of it's neighbor
        # sl = tf.slice(a, [0, 1], [tf.shape(a)[0], tf.shape(a)[1] - 1])
        # equivalent code
        sl = comp_inp[:, 2:]


        # concat with retrieved state
        inp = tf.cast(tf.concat([sl, gat], axis=1),tf.float32)

        # evaluate next state and multiply by the arch-node conversion matrix to obtain per-node states
        layer_1_state = self.layer_1_state(inp)
        layer_2_state = self.layer_2_state(layer_1_state)

        #this operation sums the states matrices for the rows in layer_2_state that correspond to the jth
        #dimension in the arcnode
        state = tf.cast(tf.sparse.sparse_dense_matmul(self.ArcNode, layer_2_state),tf.float32)
        
        if self.graph_based:
            state = tf.cast(tf.sparse.sparse_dense_matmul(self.NodeGraph, state),tf.float32)

        
        layer_1_out = self.layer_1_out(state)
        layer_2_out = self.layer_2_out(layer_1_out)
    
        return layer_2_out

    def train_step(self, comp_inp, labels,mask=None):
        
        with tf.GradientTape() as tape:
            
            output = self.call(comp_inp)
            if mask is None:
                loss_value = self.loss_fn(labels,output)
            else:
                loss_value = self.loss_fn(labels,output,mask=mask)
                
            grads = tape.gradient(loss_value, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    
        return loss_value
    
    
    def predict(self, comp_inp, ArcNode, NodeGraph=None):
        
        TrainArcNode = self.ArcNode
        
        self.ArcNode = tf.sparse.SparseTensor(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        
        if self.graph_based:
            TrainNodeGraph = self.NodeGraph
            self.NodeGraph = tf.sparse.SparseTensor(indices=NodeGraph.indices, values=NodeGraph.values,
                                        dense_shape=NodeGraph.dense_shape)
        
        
        state_init = np.zeros((ArcNode.dense_shape[0], self.state_dim))
        
        state_old_init = np.ones((ArcNode.dense_shape[0], self.state_dim))
        
        
        state = tf.Variable(state_init, name="state",trainable=False,dtype=tf.float32)

        state_old = tf.Variable(state_old_init, name="old_state",trainable=False,dtype=tf.float32)
        
        k = tf.constant(0)
        
        res, st, old_st, num = tf.while_loop(self.condition, self.convergence,
                                                 [comp_inp, state, state_old, k])
        
        gat = tf.gather(st, tf.cast(comp_inp[:, 1], tf.int32))

        
        # slice to consider only label of the node and that of it's neighbor
        # sl = tf.slice(a, [0, 1], [tf.shape(a)[0], tf.shape(a)[1] - 1])
        # equivalent code
        sl = comp_inp[:, 2:]
        

        # concat with retrieved state
        inp = tf.cast(tf.concat([sl, gat], axis=1),tf.float32)

        # evaluate next state and multiply by the arch-node conversion matrix to obtain per-node states
        layer_1_state = self.layer_1_state(inp)
        layer_2_state = self.layer_2_state(layer_1_state)

        state = tf.cast(tf.sparse.sparse_dense_matmul(self.ArcNode, layer_2_state),tf.float32)

        if self.graph_based:
            state = tf.cast(tf.sparse.sparse_dense_matmul(self.NodeGraph, state),tf.float32)
        
        layer_1_out = self.layer_1_out(state)
        layer_2_out = self.layer_2_out(layer_1_out)

        self.ArcNode = TrainArcNode
        
        if self.graph_based:
            self.NodeGraph = TrainNodeGraph
        
        return layer_2_out

@tf.custom_gradient
def gather_dense_gradient(params, indices):
    def grad(ys):
        return tf.scatter_nd(tf.expand_dims(indices, 1), ys, tf.shape(params)), None

    return tf.gather(params, indices), grad
