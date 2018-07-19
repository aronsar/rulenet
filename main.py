import tensorflow as tf
import numpy as np
import pdb

DATADIR = './new_facebook/'
USER_ID = '107' # options are 0, 107, 348, 414, ...
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
NUM_STEPS = 100 # number of learning steps; 1 step per alter
BATCH_SIZE = 940
DISPLAY_EVERY = 1

np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

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

def forwardPass(X, B, weights, biases):
  HL1 = tf.add(tf.matmul(X, weights['HL1']), biases['HL1'])
  HL1 = tf.nn.relu(HL1)
  
  BO2 = tf.add(tf.matmul(HL1, weights['BO2']), biases['BO2'])
  BO2 = tf.nn.relu(BO2)
  #BO2 = tf.add(tf.matmul(HL1, weights['BO2']), B)
  #BO2 = tf.multiply(BO2, B)
  # do one of:
    # add in logical feature vector as bias
    # multiply in the logical feature vector as bias
    # transform logical feature vector from [0,1] to [1,10] and multiply in as bias
  
  HL3 = tf.add(tf.matmul(BO2, weights['HL3']), biases['HL3'])
  
  return HL3
  
def modelPrecisionRecall(logits, labels):
  # We compute precision by dividing the number of correctly assigned circle memberships
  # by the total number of assigned circle memberships in pred. 
  
  # To find recall, we divide the total correctly assigned circle memberships 
  # by the total number of circle memberships in the labels.
  
  pred = tf.round(tf.nn.sigmoid(logits)) # basically .5 thresholding
  mult = tf.multiply(pred, labels)
  precision = tf.divide(tf.reduce_sum(mult), tf.reduce_sum(pred))
  recall = tf.divide(tf.reduce_sum(mult), tf.reduce_sum(labels))
  #recall = tf.Print(recall, [tf.reduce_sum(labels), tf.reduce_sum(pred), tf.reduce_sum(mult)])
  #recall = tf.Print(recall, [tf.reduce_min(pred), tf.reduce_max(pred), tf.reduce_min(labels), tf.reduce_max(labels), tf.reduce_min(mult), tf.reduce_max(mult)])
  return precision, recall

def main():
  print('Loading data ... ', end='') # don't print new line (for aesthetic reasons)
  X_trainm, X_testm, B_trainm, B_testm, Y_trainm, Y_testm = dataReader(DATADIR, USER_ID)
  num_alters = file_len(DATADIR + USER_ID + '.feat')
  num_feat = file_len(DATADIR + USER_ID + '.featnames')
  num_circles = file_len(DATADIR + USER_ID + '.circles')
  print('Loaded!\n')
  
  # Training
  print('Training ... ')
  graph = tf.Graph()
  
  with graph.as_default():
    # convert data to tensors
    X_train = tf.convert_to_tensor(X_trainm, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_testm, dtype=tf.float32)
    B_train = tf.convert_to_tensor(B_trainm, dtype=tf.float32)
    B_test = tf.convert_to_tensor(B_testm, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(Y_trainm, dtype=tf.float32)
    Y_test = tf.convert_to_tensor(Y_testm, dtype=tf.float32)
    n_alt = tf.convert_to_tensor(num_alters, dtype=tf.int32)

    # defining training batch
    step_ph = tf.placeholder(dtype=tf.int32, shape=())
    idx = tf.mod(step_ph*BATCH_SIZE, n_alt-BATCH_SIZE)
    X = X_train[idx:(idx+BATCH_SIZE), :]
    B = B_train[idx:(idx+BATCH_SIZE), :]
    Y = Y_train[idx:(idx+BATCH_SIZE), :]

    # hidden layer sizes
    h1 = int((num_alters + num_feat) / 2)
    
    # defining weights and biases
    weights = {
      'HL1': tf.Variable( tf.random_normal([num_alters, h1], seed=RANDOM_SEED), dtype=tf.float32),
      'BO2': tf.Variable(tf.random_normal([h1, num_feat], seed=RANDOM_SEED), dtype=tf.float32),
      'HL3': tf.Variable(tf.random_normal([num_feat, num_circles], seed=RANDOM_SEED), dtype=tf.float32),
      }

    biases = {
      'HL1': tf.Variable(tf.random_normal([h1], seed=RANDOM_SEED), dtype=tf.float32),
      'BO2': tf.Variable(tf.random_normal([num_feat], seed=RANDOM_SEED), dtype=tf.float32),
      'HL3': tf.Variable(tf.random_normal([num_circles], seed=RANDOM_SEED), dtype=tf.float32)
    }

    logits = forwardPass(X, B, weights, biases)
    
    loss = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    
    tr_logits = forwardPass(X_train, B_train, weights, biases)
    tr_precision, tr_recall = modelPrecisionRecall(tr_logits, Y_train)
    
    te_logits = forwardPass(X_test, B_test, weights, biases)
    te_precision, te_recall = modelPrecisionRecall(te_logits, Y_test)
    
  with tf.Session(graph=graph) as sess:
    # Initialize the model parameters
    tf.global_variables_initializer().run()
  
    for step in range(NUM_STEPS):
      feed_dict = {step_ph:step}

          
      if step % DISPLAY_EVERY:
        _, loss_value = sess.run([optimizer, loss], feed_dict=feed_dict)
      else:
        _, loss_value, tr_p, tr_r, te_p, te_r = sess.run(                        \
            [optimizer, loss, tr_precision, tr_recall, te_precision, te_recall], \
            feed_dict=feed_dict)
        print(('step {:4d}:   loss={:7.3f}   tr_p={:.3f}   tr_r={:.3f}'          \
             + '   te_p={:.3f}   te_r={:.3f}').format(                           \
              step, loss_value, tr_p, tr_r, te_p, te_r))
      
if __name__ == '__main__':
  main()