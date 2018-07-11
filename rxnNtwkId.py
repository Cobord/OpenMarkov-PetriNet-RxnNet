import numpy as np
import tensorflow as tf

numObs = 100
numSpecies = 10
howManyKernel = 2
howManyCatalysts = 3

# concentration data
dataAtEq = tf.placeholder(tf.float32,shape=(numObs,numSpecies))

# logarithms of concentration data
dataAtEq2 = tf.log(dataAtEq)

# subtract off the last row from everything
# multiply by a (numObs-1 \times numObs) matrix to do this
preMultiplyMatrix = tf.get_variable("preMultiply",dtype=tf.float32,initializer=tf.eye(numObs-1))
paddings = tf.constant([[0,0],[0,1]])
preMultiplyMatrix = tf.pad(preMultiplyMatrix,paddings,"CONSTANT",constant_values=-1)
dataAtEq3 = tf.matmul(preMultiplyMatrix,dataAtEq2)

# solve for the kernel over the integers with smallest l1 norm, get the howManyKernel shortest vectors
myZeros = tf.zeros([numObs,1],dtype=tf.float32)
# this just gives 0 right now, need to find the smallest NONzero solution
kernel = tf.matrix_solve_ls(dataAtEq3,myZeros)

# for each vector in the kernel, provide candidate reactions by making the basic reaction, and adding at most
# howManyCatalysts to both sides of the reaction in all possible ways, each comes with a free parameter r such that
# k_{fwd}=Kr and k_{bwd}=r, K is solved for from the data and the element of the kernel.
allCandidateRxns = 