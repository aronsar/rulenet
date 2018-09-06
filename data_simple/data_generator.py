# this script generates an npz file with n instances of d dimensional data
# the data is bounded between -1 and 1
# it also generates the label file depending on the given target function

import numpy as np
import argparse

def quadrant_binary(data_matrix):
  """
  This target function randomly assigns some quadrants to be the label True, and 
  the rest of the quadrants to be the label False. Takes the data matrix as
  argument, and returns the label vector.
  """
  
  n, d = data_matrix.shape
  true_quadrants = np.random.randint(0, 2, size=np.power(2, d))
  print("The true quadrants are:")
  list = []
  for i, quadrant in enumerate(true_quadrants):
    if quadrant == 1:
      list.append(i)
  print(list)
  
  bool_matrix = data_matrix > 0
  
  # next line assumes that each row of bool_matrix is a binary number and returns
  # a list of decimal conversions. Eg: if the 6th row of bool_matrix is [1,0,0,1],
  # then the 6th element of quadrants will be 9.
  quadrants = np.array([bool_matrix[i,:].dot(1<<np.arange(d)[::-1]) for i in range(n)])
  
  label_vector = true_quadrants[quadrants]
  
  # now we turn that label_vector of indices into a one hot label matrix
  label_matrix = np.zeros((label_vector.size, label_vector.max()+1))
  label_matrix[np.arange(label_vector.size), label_vector] = 1
  import pdb; pdb.set_trace()
  return label_matrix, bool_matrix
  
def main():
  parser = argparse.ArgumentParser(description='Generate a simple dataset')
  parser.add_argument('--n', type=int, required=True, dest='n_instances',
                      help='the number of data instances')
                      
  parser.add_argument('--d', type=int, required=True, dest='n_dimensions',
                      help='the number of data dimensions')
                      
  parser.add_argument('--data_name', type=str, dest='data_file_name',
                      help='name of the output file',
                      default='data')
                      
  parser.add_argument('--label_name', type=str, dest='label_file_name',
                      help='name of the label file',
                      default='label')
                      
  parser.add_argument('--target_function', type=str, dest='target_function',
                      help='the function that the model tries to learn ',
                      choices={'quadrant_binary', 'quadrant_multi', 'circles'},
                      default='quadrant_binary')
                      
  args = parser.parse_args()

  n = args.n_instances
  d = args.n_dimensions
  data_file_name = args.data_file_name
  label_file_name = args.label_file_name
  target_function = args.target_function

  data_matrix = np.random.rand(n, d)
  data_matrix = 2 * (data_matrix - .5) # transform to [-1, 1] range

  np.savez(data_file_name, data=data_matrix)

  if target_function == 'quadrant_binary':
    target_function = quadrant_binary
  elif target_function == 'quadrant_multi':
    target_function = quadrant_multi
  elif target_function == 'circles':
    target_function = circles
    
  label_vector, bool_matrix = target_function(data_matrix)
  np.savez(label_file_name, label=label_vector)
  np.savez('bool_mat', bool_mat=bool_matrix)
if __name__ == '__main__':
  main()