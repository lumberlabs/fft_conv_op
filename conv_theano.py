#!/usr/bin/env python

import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

if __name__ == "__main__":
    batch_size = 1
    num_images = 4
    image_dim = 4
    num_kernels = 2
    kernel_dim = 2

    filter_shape = (num_kernels, num_images, kernel_dim, kernel_dim)
    image_shape = (batch_size, num_images, image_dim, image_dim)

    images = T.matrix(dtype=theano.config.floatX)
    one_kernel = numpy.ones((kernel_dim, kernel_dim), dtype=theano.config.floatX)
    kernels = []
    for x in xrange(num_kernels):
        sub_kernels = []
        for y in xrange(num_images):
            sub_kernels.append(one_kernel * (x + 1) * (y + 1))
        kernels.append(sub_kernels)
    test_kernels = numpy.asarray(kernels)
    print "KERNELS:"
    print test_kernels
    print "-------"
    shared_kernels = theano.shared(value=test_kernels)

    reshaped_images = images.reshape((batch_size, num_images, image_dim, image_dim))
    conv_out = conv.conv2d(input=reshaped_images,
                           filters=shared_kernels, 
                           filter_shape=filter_shape,
                           image_shape=image_shape,
                           border_mode="full")
    f = theano.function(inputs=[images], outputs=conv_out)

    one_image = numpy.ones((image_dim, image_dim), dtype=theano.config.floatX)
    images = []
    for x in xrange(batch_size):
        sub_images = []
        for y in xrange(num_images):
            sub_images.append(one_image * y)
        images.append(sub_images)
    test_images = numpy.asarray(images)
    # test_images = numpy.ones((batch_size, num_images, image_dim, image_dim), dtype=theano.config.floatX)
    print "TEST IMAGES"
    print test_images
    print "----------"

    test_images = test_images.reshape(batch_size * num_images, image_dim * image_dim)

    print "RESULT (batch_size x num_kernels x convolved_rows x convolved_cols)"
    print f(test_images)
    # print f(test_images).shape
    # output shape is (batch_size, num_kernels, convolved_rows, convolved_cols)
    # for iteration in xrange(1000):
    #     f(test_images)

