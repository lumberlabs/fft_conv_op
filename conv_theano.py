#!/usr/bin/env python

import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

RUN_SPEED_TEST = True

if __name__ == "__main__":
    if RUN_SPEED_TEST:
        batch_size = 500
        num_images = 1
        image_dim = 28
        num_kernels = 50
        kernel_dim = 5
    else:
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
    if not RUN_SPEED_TEST:
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
            sub_images.append(one_image * (y + 1))
        images.append(sub_images)
    test_images = numpy.asarray(images)
    if not RUN_SPEED_TEST:
        print "TEST IMAGES"
        print test_images
        print "----------"

    test_images = test_images.reshape(batch_size * num_images, image_dim * image_dim)

    if not RUN_SPEED_TEST:
        print "RESULT (batch_size x num_kernels x convolved_rows x convolved_cols)"
        print f(test_images)

    if RUN_SPEED_TEST:
        print "batch size: {b}, num_images: {i}, image size: {ims}x{ims}, " \
              "num_kernels: {k}, kernel size: {ks}x{ks}".format(b=batch_size,
                                                                i=num_images,
                                                                ims=image_dim,
                                                                k=num_kernels,
                                                                ks=kernel_dim)
        for iteration in xrange(1000):
            f(test_images)

