import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
# # data = np.load('data2D.npy')
# data = np.load('data100D.npy')
# [num_pts, dim] = np.shape(data)
# # print(data)
# # print(num_pts)
# # print(dim)

# Distance function for K-means
def distanceFunc(X, MU):
  # Inputs
  # X: is an NxD matrix (N observations and D dimensions)
  # MU: is an KxD matrix (K means and D dimensions)
  # Outputs
  # pair_dist: is the pairwise distance matrix (NxK)
  expanded_X = tf.expand_dims(X, 0)
  expanded_MU = tf.expand_dims(MU, 1)
  distances = tf.reduce_sum(tf.square(tf.subtract(expanded_X, expanded_MU)), 2) #KxN

  return distances

def plotLoss(loss, K, is_valid=False):
  plt.plot(loss)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  if is_valid:
    plt.title('validation loss with K = {}'.format(K))
  else:
    plt.title('training loss with K = {}'.format(K))

def kMeans(K, is_valid=False):
  print('Enter K = {} -----------------------------------------'.format(K))
  # For Validation set
  train_data = data
  train_batch = num_pts
  if is_valid:
    valid_batch = int(num_pts / 3.0)
    train_batch = num_pts - valid_batch
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    train_data = data[rnd_idx[valid_batch:]]


  MU = tf.Variable(tf.random_normal([K, dim], dtype=tf.float32))
  X = tf.placeholder(tf.float32, [None, dim])

  distances = distanceFunc(X, MU)
  mins = tf.reduce_min(distances, 0)
  assign_indices = tf.argmin(distances, 0)
  loss_mu = tf.reduce_sum(mins, 0)
  adam_op = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss_mu)
  percentages = tf.unique_with_counts(assign_indices)

  init = tf.global_variables_initializer()
  sess = tf.InteractiveSession()
  sess.run(init)

  train_loss = []
  valid_loss = []
  for i in range(500):
    _, MU_value, assignment_values, loss, percentages_value = sess.run([adam_op, MU, assign_indices, loss_mu, percentages], feed_dict={X: train_data})
    train_loss.append(loss)
    if is_valid:
      val_loss = sess.run(loss_mu, feed_dict={X: val_data})
      valid_loss.append(val_loss)


  # compute percentage
  # print(np.divide(percentages_value[2:], train_batch))
  # print('final loss on train data :')
  # print(train_loss[len(train_loss) - 1])
  # print('final loss on valid data :')
  # print(valid_loss[len(valid_loss) - 1])
  print(MU_value)


  # plt.figure(figsize=(16, 4))
  # plt.subplot(121)
  # if is_valid:
  #   plotLoss(valid_loss, K, is_valid=True)
  # else:
  #   plotLoss(train_loss, K)
  #
  # plt.subplot(122)
  plt.figure(figsize=(6, 4))
  plt.scatter(train_data[:, 0], train_data[:, 1], c=assignment_values, s=1, alpha=0.5)
  plt.plot(MU_value[:, 0], MU_value[:, 1], 'r^', markersize=8)
  plt.title('K = {}'.format(K))
  # plt.savefig('images/loss_scatter_with_{}.png'.format(K))
  # plt.savefig('images/val_loss_scatter_with_{}.png'.format(K))
  plt.savefig('images/Kmean_compare_scatter_with_{}.png'.format(K))


# kMeans(1)
# kMeans(2)
# kMeans(3)
# kMeans(4)
# kMeans(5)

# remember to correct the plot title!!!!!!!!!!!!
# kMeans(1, is_valid=True)
# kMeans(2, is_valid=True)
# kMeans(3, is_valid=True)
# kMeans(4, is_valid=True)
# kMeans(5, is_valid=True)

# last part!!!!!!!!!!!!!
kMeans(5)
kMeans(10)
kMeans(15)
kMeans(20)
kMeans(30)
plt.show()
