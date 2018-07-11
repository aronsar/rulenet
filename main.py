import tensorflow as tf
import numpy as np
import pdb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

DATADIR = './new_facebook/'
USER_ID = '0'

class Alter:
  def __init__(self, feature_vec, mutual_friends_vec, circle_membership):
    self.feature_vec = feature_vec
    self.mutual_friends_vec = mutual_friends_vec
    self.circle_membership = circle_membership

    
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
        circle_membership_matrix[int(member),circle_index] = 1
  
  # the following for loop iterates over row vectors of two matrices
  for mutual_friend in range(num_mutual_friends):
    alter_array.append(Alter(feature_matrix[mutual_friend, :],
                             mutual_friends_matrix[mutual_friend, :],
                             circle_membership_matrix[mutual_friend, :]))
                             
  shuffled_alter_array = np.random.permutation(alter_array)
  lim = int(.9 * num_mutual_friends)
  train_alters = shuffled_alter_array[:lim]
  test_alters = shuffled_alter_array[lim:]
  
  return train_alters, test_alters



def main():
  train_alters, test_alters = dataReader(DATADIR, USER_ID)

if __name__ == '__main__':
    main()