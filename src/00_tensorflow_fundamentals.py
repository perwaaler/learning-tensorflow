# %% explore the fundamentals of tensorflow
# * introduction to tensors
# * Getting information from tensors
# * ...
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf

print(tf.__version__)

# %% creating tensors with tf.constant
scalar = tf.constant(7)
scalar.ndim  # check number of dimensions

# %% check number of dimensions
vector = tf.constant([10, 10])
vector.ndim

# %% create a matrix
matrix = tf.constant([[10, 10], [10, 10]])
matrix.ndim

# create a tensor with specified datatype:
another_matrix = tf.constant([[10., 7.], [3., 2.], [1., 3.]], dtype=tf.float16)
another_matrix = tf.constant([[10., 7], [3, 2], [1, 3]])
another_matrix.ndim
# this is done to save space (default is 32 bit)

# %% creating a tensor (3-dim matrix):
tensor = tf.constant([[[1, 1], [2, 2], [3, 3]],
                      [[1, 1], [2, 2], [3, 3]]])
tensor.ndim

# %% use "tf.Variable" to create a changeable tensor:
unchangeable_tensor = tf.constant([1, 2])
changeable_tensor = tf.Variable([1, 2], dtype=tf.float16)

# change one of the variables in the changeable tensor
# unchangeable_tensor[0] = 7 does not work!
changeable_tensor[0].assign(7.)
# %% create random tensors
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3, 2), mean=5, stddev=1)
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3, 2), mean=5, stddev=1)


# are they equal?
random_1, random_2, random_1 == random_2

# %% shuffle the order of elements in a tensor
not_shuffled = tf.constant([[1, 2],
                           [1, 2],
                           [3, 4]])
not_shuffled

# %% shuffle tensor
tf.random.set_seed(1)
shuffled = tf.random.shuffle(not_shuffled, seed=3)
shuffled


# %% create a tensor of all ones
tf.ones([10, 7, 2])
tf.zeros([10, 7, 2])

# %% you can convert numpy arrays into tensors:
numpy_A = np.arange(1, 25, dtype=np.int32)
numpy_A = tf.constant(numpy_A, shape=(2, 2, 6))
numpy_A


# %% getting information from tensors
numpy_A.ndim
numpy_A.shape
sz = tf.size(numpy_A).numpy()
tf.shape(numpy_A)
sz

# tensor properties:
# * shape
# * size
# * dimension
# * axis


# %% index tensors ----------------------------------------------------
A = tf.Variable([[[1, 2],
                  [3, 5]],
                 [[6, 1],
                  [4, 6]],
                 [[3, 1],
                  [9, 2]]], dtype=tf.float32)

B = A[:1, :, -1]

# %% reshape a tensor
rank_2_tensor = tf.constant([[10, 7],
                             [3, 4]])

# %% expand dimensions
rank_3_tensors = rank_2_tensor[:, :, tf.newaxis]
# alternative:
# rank_3_tensors = rank_2_tensor[..., tf.newaxis]
B = tf.expand_dims(rank_2_tensor, axis=0)
C = tf.expand_dims(rank_2_tensor, axis=1)
D = tf.expand_dims(rank_2_tensor, axis=2)
B[:, 1, 0]


# %% Manipulating Tensors ---------------------------------------------
# ¤¤¤ Basic operations ¤¤¤
# +, -, *, /

tensor = tf.constant([[10, 7],
                      [3, 4]], dtype=tf.float32)
# can be multiplied by scalars:
tensor = tensor + 10
tensor = tensor * 0.25
# two ways of exponentiating:
tensor = tf.exp(tensor)
# or use the tensorflow math library:
tf.math.exp(tensor)

# %% the tensor flow version of doing things is often faster
tensor = tf.constant([[10, 7],
                     [3, 4]], dtype=tf.float32)

# some different operations in tensorflow
tensor = tf.multiply(tensor, 0.25)
tensor = tf.add(tensor, 10)
tensor = tf.math.log(tensor)
tensor = tf.sin(tensor)

print(tensor)

# %% matrix multiplication in tensorflow ¤¤
tf.matmul(tensor, tensor)
# elementwise multiplication:
tensor*tensor

# matrix multiplication using "@" ¤¤
tensor @ tensor

# transpose matrix
tf.transpose(tensor)

# %% in order for multiplication to be defined, inner dimensions must match . This # can be done either by reshaping or by transposing.
tf.reshape(tensor, shape=(4, 1))

# %% tensor multiplication using tensordot ¤¤
X = tf.constant([[2, 1],
                 [3, 9]])
Y = tf.constant([[4, 1, 2],
                 [1, 2, 3]])
tf.tensordot(X, Y, axes=0)
tf.tensordot(X, Y, axes=1)

# getting a bit silly:
tf.matmul(X, tf.reshape(tf.transpose(Y), shape=(2, 3)))

# %% Take dot product of two tensor vectors:
a = tf.constant([1, 2, 3])
b = tf.constant([1, 2, 3])
c = tf.tensordot(a, b, axes=1)
print(f"the dot product of a and b is {c}")

# %% investigate silent "errors" by printing various values:
print(f"original Y is:\n{Y}"
      f"Y transposed is:\n{tf.transpose(Y)}"
      f"Y reshaped to (3,2) is:\n{tf.reshape(Y, shape=(3,2))}")


# %% ¤¤¤ Changing Datatypes - Casting ¤¤¤
X = tf.constant([1., 2., 3.], dtype=tf.float32)
print(f"matrix is: {X}\n"
      f"it has datatype:{X.dtype}")

# %% changing ("casting") from 32 bit memory to 16 bit memory:
X = tf.constant([1., 2., 3.], dtype=tf.float32)
X = tf.cast(X, dtype=tf.float16)
print(f"datatype is now:{X.dtype}")

# X can only be multiplied with object of same datatype:
Y = tf.constant([1., 2., 3.], dtype=tf.float16)
Z = tf.tensordot(X, Y, axes=1)

print(f"X is \n{X}\nY is \n{Y}\n the dot product is {Z}")


# %% Aggregating tensors (abs, min, max, sum, etc...):
X_det = tf.constant([-1., 2., 3.])
# generate normal random values using tensorflow:
X_rand0 = tf.random.Generator.from_seed(1).normal(mean=0,
                                                  stddev=1.2,
                                                  shape=(5, 1))
# generate uniformly distributed integers using tensorflow:
X_rand_int_mat = tf.random.Generator.from_seed(1).uniform(shape=(2, 3),
                                                          minval=1,
                                                          maxval=10,
                                                          dtype=tf.int32)
# alternative way:
# tf.random.set_seed(1)
# X_rand_int_mat = tf.random.uniform(shape=(2,3), minval=1, maxval=5, dtype=tf.int32)
print(X_rand_int_mat)
# generate uniformly distributed integers using numpy.rand:
X_rand1 = tf.constant(np.random.randint(low=1,
                                        high=50,
                                        size=10))

X = X_rand0
abs = tf.abs(X)
min = tf.reduce_min(X, axis=0)
max = tf.reduce_max(X, axis=0)
mean = tf.reduce_mean(X)
sum = tf.reduce_sum(X)

# var = tf.math.reduce_variance(X)
var = tfp.stats.variance(X)
# std = tf.math.reduce_std(X)
std = tfp.stats.stddev(X)


# NOTE: tensorflow tends to put reduce_ in front of its aggregation methods.

print(f"the elements of X are {X}\n"
      f"the absolute value of the elements of X is {abs}\n"
      f"the mean of X is {mean}\n"
      f"the minimum of X is {min}\n"
      f"the maximum of X is {max}\n"
      f"the sum of X is {sum}\n"
      f"the variance of X is {var}\n"
      f"the standard deviation of X is {std}")


# %% Find the positional maximum and minimum
X = tf.Variable([1, 2, 3, 2, 5, 1])
X[0].assign(3)
argmax = tf.math.argmax(X)
print(f"X is {X.numpy()}\nthe positional maximum is {argmax}")

X_mat = tf.Variable([[2.1, 2., -1.],
                     [1., 3.2, 2.]])
# NOTE: argmax by default reduces over axis=0, i.e. it takes maximum columnwise
argmax = tf.math.argmax(X_mat)
argmin = tf.math.argmin(X_mat)
# evaluate at maximum positions
col_ind = tf.range(X_mat.shape[1])
maxvals = X_mat.numpy()[argmax, col_ind]
minvals = X_mat.numpy()[argmin, col_ind]
maxval = tf.reduce_max(maxvals)
minval = tf.reduce_min(minvals)


print(f"X is \n{X_mat.numpy()}\n"
      f"the positional maximum of each column are {argmax}\n"
      f"the positional minimum of each column are {argmin}\n"
      f"where it takes max values {maxvals}\n"
      f"where it takes min values {minvals}\n"
      f"the maximum value of X is {maxval}\n"
      f"the minimum value of X is {minval}")


assert maxval == tf.reduce_max(X_mat)


# %% squeezing a tensor
G = tf.Variable(tf.random.uniform(shape=[50]), dtype=tf.float32)
G.shape
G_squeezed = tf.squeeze(G)
G_squeezed.shape


# %% ¤¤¤ one hot encoding ¤¤¤
x = [0, 1, 2, 3, 1]
x_onehot = tf.one_hot(x, depth=4)
# specify costum values for one-hot-encoding:
x_onehot_relabeled = tf.one_hot(x,
                                depth=len(x),
                                on_value="yes",
                                off_value="no")

print(f"the one hot encoding of {x} is \n"
      f"{x_onehot}\n"
      f"After relabeling the encoded version of x is:\n"
      f"{x_onehot_relabeled}")

# %%  some operations (sqrt, log, etc...)
x = tf.range(10)
x = tf.square(x)
# for some operations, integers are not allowed as inputs:
x_sqrt = tf.sqrt(tf.cast(x, dtype=tf.float32))

# %% convert between numpy and tensorflow type:
x = tf.range(2)
# tf ----> np:
x = x.numpy()
print(x)
# np ---> tf:
x = tf.constant(x)
print(x)

# %% be vary of default datatypes!
x = tf.constant([1., 2.])
y = tf.constant(np.array([1., 2.]))
print(x.dtype, y.dtype)

