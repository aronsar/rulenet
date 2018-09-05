import tensorflow as tf
import numpy as np
from subprocess import call
import pdb

DATADIR = './new_facebook/'
USER_ID = '107' # options are 0, 107, 348, 414, ...
LEARNING_RATE = 1e-3
LAMBDA = 0.01 # regularization constant
RANDOM_SEED = 42
NUM_STEPS = 800 # number of learning steps
BATCH_SIZE = 940# for ego 107, 940 is all the training alters. Try reducing BATCH_SIZE to just 10 or 50
DISPLAY_EVERY = 1

np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

call("rm -rf tf_logs")

def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

def dataReader(data_dir=DATADIR, user_id=USER_ID):
  num_mutual_friends = file_len(DATADIR + USER_ID + '.feat')
  num_feat = file_len(DATADIR + USER_ID + '.featnames')
  num_circles = file_len(DATADIR + USER_ID + '.circles')
  
  mutual_friends_matrix = np.zeros([num_mutual_friends, num_mutual_friends]) # binary friend matrix (sparse)
  feature_matrix = np.zeros([num_mutual_friends, num_feat])
  circle_membership_matrix = np.zeros([num_mutual_friends, num_circles])
  alter_array = []
  
  # obtain the edges/mutual friends, and populate alter_array
  with open(DATADIR + USER_ID + '.edges') as f:
    for line in f:
      edge_idxs = [int(x) for x in line.split(' ')] # tuple of indices
      mutual_friends_matrix[edge_idxs[0], edge_idxs[1]] = 1
  
  # get the features, and populate feature_matrix
  with open(DATADIR + USER_ID + '.feat') as f:
    for i, line in enumerate(f):
      feature_vec = [int(x) for x in line.split(' ')]
      feature_matrix[i,:] = feature_vec[1:] # dropping the first number, which is basically just the line number
  
  # get the circle membership info, and populate the circle membership matrix
  with open(DATADIR + USER_ID + '.circles') as f:
    for circle_index, line in enumerate(f):
      for member in line.split(' ')[1:]: # we ignore the first word, which is just the circle index
        if not member.isspace(): circle_membership_matrix[int(member),circle_index] = 1
  
  idxs = np.random.shuffle(np.arange(num_mutual_friends))
  lim = int(.9 * num_mutual_friends)
  X_trainm = mutual_friends_matrix[:lim,:] # the m at the end of X_trainm is for "matrix"
  X_testm = mutual_friends_matrix[lim:,:]
  B_trainm = feature_matrix[:lim,:]
  B_testm = feature_matrix[lim:,:]
  Y_trainm = circle_membership_matrix[:lim,:]
  Y_testm = circle_membership_matrix[lim:,:]
  
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
    mean, variance = tf.nn.moments(preac, axes=0)
    preac = (preac - mean) / tf.sqrt(variance + 1.0e-6)
    tf.summary.histogram('pre_activations', preac)
    activations = act(preac, name='activation')
    tf.summary.histogram('activations', activations)
    with_bool = activations + boolean_tensor
    tf.summary.histogram('with_bool', with_bool)
    return with_bool
    
def forwardPass(X, B, num_alters, num_feat, num_circles):
  # hidden layer sizes
  # FIXME: insert summary of what function does (docstring comment)
  h1 = int((num_alters + num_feat) / 2)
  #h1 = 30
  layer1 = affine_layer(X, num_alters, h1, 'affine_layer1')
  layer2 = bool_injection_layer(layer1, B, h1, num_feat, 'bool_inj_layer2')
  layer3 = bool_injection_layer(layer2, B, num_feat, num_feat, 'bool_inj_layer3')
  logits = affine_layer(layer3, num_feat, num_circles, 'affine_layer4', act=tf.identity)
  
  return logits
  
def modelPrecisionRecall(logits, labels):
  # We compute precision by dividing the number of correctly assigned circle memberships
  # by the total number of assigned circle memberships in pred. 
  
  # To find recall, we divide the total correctly assigned circle memberships 
  # by the total number of circle memberships in the labels.
  
  pred = tf.round(tf.nn.sigmoid(logits)) # maps from [-inf, inf] to {0, 1}
  mult = tf.multiply(pred, labels)
  precision = tf.divide(tf.reduce_sum(mult), tf.reduce_sum(pred))
  recall = tf.divide(tf.reduce_sum(mult), tf.reduce_sum(labels))
  #recall = tf.Print(recall, [tf.reduce_sum(labels), tf.reduce_sum(pred), tf.reduce_sum(mult)])
  #recall = tf.Print(recall, [tf.reduce_min(pred), tf.reduce_max(pred), tf.reduce_min(labels), tf.reduce_max(labels), tf.reduce_min(mult), tf.reduce_max(mult)])
  return precision, recall

def main():
  print('Loading data ... ', end='') # end='' doesn't print new line for aesthetic reasons
  X_trainm, X_testm, B_trainm, B_testm, Y_trainm, Y_testm = dataReader(DATADIR, USER_ID)
  num_alters = file_len(DATADIR + USER_ID + '.feat')
  num_feat = file_len(DATADIR + USER_ID + '.featnames')
  num_circles = file_len(DATADIR + USER_ID + '.circles')
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
    n_alt = tf.convert_to_tensor(num_alters, dtype=tf.int32) #FIXME: try removing n_alt

    # assemble batches
    step_ph = tf.placeholder(dtype=tf.int32, shape=())
    idx = tf.mod(step_ph*BATCH_SIZE, n_alt-BATCH_SIZE)
    X = X_train[idx:(idx+BATCH_SIZE), :]
    B = B_train[idx:(idx+BATCH_SIZE), :]
    Y = Y_train[idx:(idx+BATCH_SIZE), :]
    
    # compute loss and define optimizer
    logits = forwardPass(X, B, num_alters, num_feat, num_circles)
    loss = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=Y) \
               + tf.losses.get_regularization_loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    
    # precision and recall
    tr_logits = forwardPass(X_train, B_train, num_alters, num_feat, num_circles)
    tr_precision, tr_recall = modelPrecisionRecall(tr_logits, Y_train)
    
    te_logits = forwardPass(X_test, B_test, num_alters, num_feat, num_circles)
    te_precision, te_recall = modelPrecisionRecall(te_logits, Y_test)
    
    # summary stuff for tensorboard
    with tf.variable_scope('prec_rec_summary'):
      tf.summary.scalar('train_precision', tr_precision)
      tf.summary.scalar('train_recall', tr_recall)
      tf.summary.scalar('test_precision', te_precision)
      tf.summary.scalar('test_recall', te_recall)
    
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
  
  # running session
  with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    summary_writer = tf.summary.FileWriter('./tf_logs', sess.graph)
    
    for step in range(NUM_STEPS):
      feed_dict = {step_ph:step}
          
      if step % DISPLAY_EVERY:
        summary, _, loss_value = sess.run([merged, optimizer, loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)
      else:
        summary, _, loss_value, tr_p, tr_r, te_p, te_r = sess.run(               \
            [merged, optimizer, loss, tr_precision, tr_recall, te_precision, te_recall], \
            feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)
        print(('step {:4d}:   loss={:7.3f}   tr_p={:.3f}   tr_r={:.3f}'          \
             + '   te_p={:.3f}   te_r={:.3f}').format(                           \
             step, loss_value, tr_p, tr_r, te_p, te_r))
               
if __name__ == '__main__':
  main()