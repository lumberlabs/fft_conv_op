
all: cufft_sample_speed cufft_sample_debug

cufft_sample_speed: cufft_sample.cu
	nvcc -o cufft_sample_speed cufft_sample.cu -lcufft -DRUN_SPEED_TESTS=1
cufft_sample_debug: cufft_sample.cu
	nvcc -o cufft_sample_debug cufft_sample.cu -lcufft -DDEBUG=1
time: time_theano time_cufft

time_theano: time_theano_orig time_theano_fft

time_theano_orig:
	THEANO_FLAGS=${THEANO_FLAGS},floatX=float32,device=gpu,mode=ProfileMode python conv_theano.py | grep GpuConv
time_theano_fft:
	THEANO_FLAGS=${THEANO_FLAGS},floatX=float32,device=gpu,mode=ProfileMode python conv_theano.py --fft| grep GpuFFTConvOp
time_cufft: cufft_sample_speed
	./cufft_sample_speed
clean:
	rm -f core.* a.out *~ *.pyc