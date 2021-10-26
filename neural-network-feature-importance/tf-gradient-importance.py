def gradient_importance(inp,model):
    inp = list(map(lambda x:tf.Variable(x,dtype=tf.float32),inp))
    with tf.GradientTape() as tape:
        predictions = model(inp)
        grads = tape.gradients(predictions, inp)
    return grads

grads = gradient_importance([input_0.astype(np.float32),input_1.astype(np.float32)])

#importance for input_0
np.mean(grads[0].numpy(),axis=0)

#importance for input_1
np.mean(grads[1].numpy(),axis=0)
