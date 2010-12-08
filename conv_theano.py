#!/usr/bin/env python

import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

if __name__ == "__main__":
    batch_size = 1
    num_images = 50
    image_dim = 8
    num_kernels = 20
    kernel_dim = 5

    filter_shape = (num_kernels, num_images, kernel_dim, kernel_dim)
    image_shape = (batch_size, num_images, image_dim, image_dim)

    images = T.matrix(dtype=theano.config.floatX)
    kernels = numpy.ones(filter_shape, dtype=theano.config.floatX)
    shared_kernels = theano.shared(value=kernels)

    reshaped_images = images.reshape((batch_size, num_images, image_dim, image_dim))
    conv_out = conv.conv2d(input=reshaped_images,
                           filters=shared_kernels, 
                           filter_shape=filter_shape,
                           image_shape=image_shape,
                           border_mode="full")
    f = theano.function(inputs=[images], outputs=[conv_out])

    test_images = numpy.ones((batch_size, num_images, image_dim, image_dim), dtype=theano.config.floatX).reshape(batch_size * num_images, image_dim * image_dim)

    for iteration in xrange(10000):
        f(test_images)

