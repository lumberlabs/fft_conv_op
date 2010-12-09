
all: cufft_sample_speed cufft_sample_debug

cufft_sample_speed: cufft_sample.cu
	nvcc -o cufft_sample_speed cufft_sample.cu -lcufft -DRUN_SPEED_TESTS=1
cufft_sample_debug: cufft_sample.cu
	nvcc -o cufft_sample_debug cufft_sample.cu -lcufft -DDEBUG=1
time_theano:
	THEANO_FLAGS=$THEANO_FLAGS,floatX=float32,device=gpu,mode=ProfileMode python conv_theano.py | grep GpuConv

clean:
	rm -f core.*