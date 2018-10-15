import tensorflow as tf
import numpy as np
from subprocess import call
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATADIR = './data_simple'
LEARNING_RATE = 2e-3
LAMBDA = 0.00000001 # regularization constant
RANDOM_SEED = 37
NUM_STEPS = 201 # number of learning steps
BATCH_SIZE = 899# 
DISPLAY_EVERY = 100

np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

call("rm -rf tf_logs")

def dataReader(data_dir=DATADIR):
  data = np.load(data_dir + '/data.npz')['data']
  bool_mat = np.load(data_dir + '/bool_mat.npz')['bool_mat']
  label = np.load(data_dir + '/label.npz')['label']
  
  lim = int(.9 * len(label))
  X_trainm = data[:lim,:] # the m at the end of X_trainm is for "matrix"
  X_testm = data[lim:,:]
  B_trainm = bool_mat[:lim,:]
  B_testm = bool_mat[lim:,:]
  Y_trainm = label[:lim,:]
  Y_testm = label[lim:,:]
  
  return X_trainm, X_testm, B_trainm, B_testm, Y_trainm, Y_testm
  
def variable_summaries(var):
  '''Attach a lot of summaries to a Tensor (for TensorBoard visualization).'''
  with tf.variable_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.variable_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def affine_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
    weights = tf.get_variable('weights', [input_dim, output_dim], initializer=\
        tf.random_normal_initializer(mean=0.0, stddev=.1, seed=RANDOM_SEED),  \
        regularizer=tf.contrib.layers.l1_regularizer(LAMBDA))
    variable_summaries(weights)
    
    biases = tf.get_variable('biases', [output_dim], initializer=             \
        tf.constant_initializer(value=0.0))
    variable_summaries(biases)
    
    preactivate = tf.matmul(input_tensor, weights) + biases
    tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def bool_injection_layer(input_tensor, boolean_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
    weights = tf.get_variable('weights', [input_dim, output_dim], initializer=\
        tf.random_normal_initializer(mean=0.0, stddev=.1, seed=RANDOM_SEED),  \
        regularizer=tf.contrib.layers.l1_regularizer(LAMBDA))
    variable_summaries(weights)
    tf.summary.image('bil_weights', tf.expand_dims(tf.expand_dims(weights, axis=-1), axis=0))
    
    biases = tf.get_variable('biases', [output_dim], initializer=             \
        tf.constant_initializer(value=0.0))
    variable_summaries(biases)
    
    preac = tf.matmul(input_tensor, weights) + biases
    #mean, variance = tf.nn.moments(preac, axes=0)
    #preac = (preac - mean) / tf.sqrt(variance + 1.0e-6) # batch normalization
    tf.summary.histogram('pre_activations', preac)
    activations = act(preac, name='activation')
    tf.summary.histogram('activations', activations)
    activations = activations * boolean_tensor
    tf.summary.histogram('with_bool', activations)
    return activations, weights
    
def forwardPass(X, B, input_dim, bool_dim, label_dim):
  # hidden layer sizes
  # FIXME: insert summary of what function does (docstring comment)
  # FIXME: also combine with display_activations if you want
  h1 = int((input_dim + bool_dim*2) / 2)
  layer1 = affine_layer(X, input_dim, h1, 'input_affine_layer')
  layer2, _ = bool_injection_layer(layer1, B, h1, bool_dim, 'first_bool_inj_layer')
  layer3, _ = bool_injection_layer(layer2, B, bool_dim, bool_dim, 'second_bool_inj_layer')
  logits = affine_layer(layer3, bool_dim, label_dim, 'last_layer', act=tf.identity)
  
  return logits
  
def forwardPass_dispAct(X, B, input_dim, bool_dim, label_dim):
  """
  Does a forward pass using the same weights as were trained in forwardPass()
  and displays the input layer, every activation layer, and the output layer.
  Also picks a random datapoint from the input datapoints for display.
  
  X: tensor of shape [batch size, input dim]
  input_dim: dimensionality of the input
  bool_dim: dimensionality of the boolean vector (2*input_dim)
  label_dim: number of possible labels
  """
  h1 = int((input_dim + bool_dim*2) / 2)
  
  #input = tf.py_func(pick_rand, [X], tf.float32)
  # FIXME: use the entire test set, not just one point
  #X = X[0:3,:]
  #B = B[0:3,:]
  layer1 = affine_layer(X, input_dim, h1, 'input_affine_layer')
  layer2, weights2 = bool_injection_layer(layer1, B, h1, bool_dim, 'first_bool_inj_layer')
  layer3, weights3 = bool_injection_layer(layer2, B, bool_dim, bool_dim, 'second_bool_inj_layer')
  logits = affine_layer(layer3, bool_dim, label_dim, 'last_layer', act=tf.identity)
  
  #true = tf.py_func(display_activations, [X, layer1, layer2, layer3, logits], tf.bool)
  hist = tf.py_func(display_histogram, [layer3, weights3], tf.float32)
  
  return hist
  
def display_histogram(layers, weights):
  # layer is (batch_size, bool_dim)
  # weights is (bool_dim, bool_dim)
  
  total_hist = np.zeros((6,6)) #FIXME: no hard code 6
  #import pdb; pdb.set_trace()
  
  # iterates over rows of the layers matrix
  for layer in layers:
    tiled_layer = np.tile(np.expand_dims(layer, axis=1), 6)
    #print(tiled_layer)
    weights_dot_act = np.dot(tiled_layer, weights)
    max_tuple = np.unravel_index(weights_dot_act.argmax(), weights_dot_act.shape)
    #print(max_tuple)
    total_hist[max_tuple] += 1
    
  
  return total_hist.astype(np.float32)
  
  
def display_activations(*args):
  for layer in args:
    print([('%.3f' % x).rjust(6,' ') for x in layer[0]])
  
  return True
  
def pick_rand(X):
  # FIXME: actually pick randomly
  return X[0,:]
  
def accuracy(logits, labels):
  """
  Calculates the accuracy the model achieves.
  
  logits: tensor of shape [batch_size, label_dim]
  labels: tensor of shape [batch_size]
  """
  
  pred = tf.argmax(input=logits, axis=1)
  labels = tf.argmax(input=labels, axis=1)
  accuracy = tf.contrib.metrics.accuracy(predictions=pred, labels=labels, name="accuracy")
  return accuracy
  
def main():
  print('Loading data ... ', end='') # end='' doesn't print new line for aesthetic reasons
  X_trainm, X_testm, B_trainm, B_testm, Y_trainm, Y_testm = dataReader(DATADIR)
  X_test_truem = X_testm[Y_testm[:,1] > 0, :]
  X_test_falsem = X_testm[Y_testm[:,1] == 0, :]
  B_test_truem = B_testm[Y_testm[:,1] > 0, :]
  B_test_falsem = B_testm[Y_testm[:,1] == 0, :]
  n_train, data_dim = X_trainm.shape
  n_test = X_testm.shape[0]
  label_dim = Y_trainm.shape[1]
  bool_dim = B_trainm.shape[1]
  print('Loaded!\n')
  
  print('Training ... ')
  
  # defining computational graph
  graph = tf.Graph()
  
  with graph.as_default():
    # convert data to tensors
    X_train = tf.convert_to_tensor(X_trainm, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_testm, dtype=tf.float32)
    B_train = tf.convert_to_tensor(B_trainm, dtype=tf.float32)
    B_test = tf.convert_to_tensor(B_testm, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(Y_trainm, dtype=tf.float32)
    Y_test = tf.convert_to_tensor(Y_testm, dtype=tf.float32)
    n_tr = tf.convert_to_tensor(n_train, dtype=tf.int32) #FIXME: try removing n_alt

    # partition test set
    X_test_true = tf.convert_to_tensor(X_test_truem, dtype=tf.float32)
    X_test_false = tf.convert_to_tensor(X_test_falsem, dtype=tf.float32)
    B_test_true = tf.convert_to_tensor(B_test_truem, dtype=tf.float32)
    B_test_false = tf.convert_to_tensor(B_test_falsem, dtype=tf.float32) 
    
    # assemble batches
    step_ph = tf.placeholder(dtype=tf.int32, shape=())
    idx = tf.mod(step_ph*BATCH_SIZE, n_tr-BATCH_SIZE)
    X = X_train[idx:(idx+BATCH_SIZE), :]
    B = B_train[idx:(idx+BATCH_SIZE), :]
    Y = Y_train[idx:(idx+BATCH_SIZE), :]
    
    # compute loss and define optimizer
    logits = forwardPass(X, B, data_dim, bool_dim, label_dim)
    #loss = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=Y) \
    #           + tf.losses.get_regularization_loss()
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y)      \
               + tf.losses.get_regularization_loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    
    # precision and recall
    tr_logits = forwardPass(X_train, B_train, data_dim, bool_dim, label_dim)
    tr_accuracy = accuracy(tr_logits, Y_train)
    
    te_logits = forwardPass(X_test, B_test, data_dim, bool_dim, label_dim)
    te_accuracy = accuracy(te_logits, Y_test)
    
    # displaying activations
    acts_hist_true = forwardPass_dispAct(X_test_true, B_test_true, data_dim, bool_dim, label_dim)
    acts_hist_false = forwardPass_dispAct(X_test_false, B_test_false, data_dim, bool_dim, label_dim)
    
    # summary stuff for tensorboard
    with tf.variable_scope('accuracy_summary'):
      tf.summary.scalar('train_accuracy', tr_accuracy)
      tf.summary.scalar('test_accuracy', te_accuracy)
    
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
  
  # running session
  with tf.Session(graph=graph) as sess:
    total_hist_true = np.zeros((6,6))
    total_hist_false = np.zeros((6,6))
    
    for r_seed in range(1000):
      np.random.seed(r_seed)
      tf.set_random_seed(r_seed)
      
      tf.global_variables_initializer().run()
      summary_writer = tf.summary.FileWriter('./tf_logs', sess.graph)
      
      for step in range(NUM_STEPS):
        feed_dict = {step_ph:step}
            
        if step % DISPLAY_EVERY:
          summary, _, loss_value = sess.run([merged, optimizer, loss], feed_dict=feed_dict)
          summary_writer.add_summary(summary, step)
        else:
          summary, _, loss_value, tr_acc, te_acc = sess.run(               \
              [merged, optimizer, loss, tr_accuracy, te_accuracy], \
              feed_dict=feed_dict)
          summary_writer.add_summary(summary, step)
          print(('step {:4d}:   loss={:7.3f}   tr_acc={:.3f}'          \
               + '   te_acc={:.3f}').format(                           \
               step, loss_value, tr_acc, te_acc))
      
      summary, ht, hf = sess.run([merged, acts_hist_true, acts_hist_false], \
          feed_dict={step_ph:1})
      
      total_hist_true += ht
      total_hist_false += hf
      
      np.set_printoptions(suppress=True)
      print(total_hist_true)
      print(total_hist_false)
      
if __name__ == '__main__':
  main()