#!/usr/bin/env python
import sys

import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

RUN_SPEED_TEST = True

if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]=='fft':
        from fft_conv_op import GpuFFTConvOp


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

    one_kernel = numpy.ones((kernel_dim, kernel_dim), dtype=theano.config.floatX)
    kernels = []
    for x in xrange(num_kernels):
        sub_kernels = []
        for y in xrange(num_images):
            sub_kernels.append(one_kernel * (x + 1) * (y + 1))
        kernels.append(sub_kernels)
    test_kernels = numpy.asarray(kernels)

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
        print "KERNELS:"
        print test_kernels
        print "-------"
    shared_kernels = theano.shared(value=test_kernels)
    shared_images = theano.shared(value=test_images)
    shared_output = theano.shared(numpy.zeros((2,3,4,5),dtype=theano.config.floatX))
    reshaped_images = shared_images.reshape((batch_size, num_images, image_dim, image_dim))
    conv_out = conv.conv2d(input=reshaped_images,
                           filters=shared_kernels, 
                           filter_shape=filter_shape,
                           image_shape=image_shape,
                           border_mode="full")
    f = theano.function(inputs=[], updates={shared_output:conv_out})
    topo = f.maker.env.toposort()
    if any([node.op.__class__.__name__=="ConvOp" for node in topo]):
        print "use CPU"
    elif any([node.op.__class__.__name__=="GpuConv" for node in topo]):
        print "use GPUConv"
    elif any([node.op.__class__.__name__=="GpuFFTConvOp" for node in topo]):
        print "use GpuFFTConvOp"
    else:
        import pdb;pdb.set_trace()
        print "use unknow"

    if not RUN_SPEED_TEST:
        print "RESULT (batch_size x num_kernels x convolved_rows x convolved_cols)"
        print f(test_images)

    if RUN_SPEED_TEST:
        print "batch size: %(batch_size)s, num_images: %(num_images)s, image size: %(image_dim)sx%(image_dim)s, " \
              "num_kernels: %(num_kernels)s, kernel size: %(kernel_dim)sx%(kernel_dim)s"%locals()
        for iteration in xrange(1000):
            f()

