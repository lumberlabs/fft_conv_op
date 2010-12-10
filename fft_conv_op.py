"""
WARNING: GpuFFTConvOp currently don't return the good answer

TODO: create the plan in the c_support_code() fct when we know the shape?
TODO: reuse preallocated memory for intermediate result?
TODO: reuse the op own preallocated memory for next intermediate result?
TODO: speed test more case including all case in the last scipy paper.
TODO: extend to cover more case, as in many case we will crash!
"""


from theano.gof import Apply, Op
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, CudaNdarrayType

print "\n\n\n WARNING: CURRENT VERSION of GpuFFTConvOp is not well optimized! \n\n\n"

class GpuFFTConvOp(Op):
    __attrnames = ['out_mode', 'check', 'debug']

    def __init__(self, output_mode='valid', check=False, debug=False,
                 more_memory=True):
        """
        :param more_memory: if True, we will keep the fft plan between each call
                            This use more memory but is faster
                            Their is no way to my knowledge to recover this 
                            memory after we finished using this fct.
        """
        self.out_mode = output_mode
        self.check=check
        self.debug=debug
        self.more_memory = more_memory
        if self.out_mode!='full':
            import pdb;pdb.set_trace()
            raise Exception(self.__class__.__name__+" support only the full mode for now")
        self._rehash()

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        for a in self.__attrnames:
            if getattr(self, a) != getattr(other, a):
                return False
        return True

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._rehash()

    def _rehash(self):
        hashval = hash(type(self))
        for a in self.__attrnames:
            hashval = hashval ^ hash(getattr(self, a))
        self.__hashval = hashval

    def __hash__(self):
        return self.__hashval

    def __str__(self):
        return self.__class__.__name__+"{" +",".join(str((a, getattr(self, a))) for a in self.__attrnames)  + "}"

    def make_node(self, inputs, kerns):
        """
        inputs - 4 dim: batches x stacksize x rows x cols
        kerns - 4 dim: nkern x stackidx x rows x cols
        """
        outdim = kerns.ndim
        _inputs = as_cuda_ndarray_variable(inputs)
        _kerns = as_cuda_ndarray_variable(kerns)
        # TODO: lift this restriction by upcasting either inputs or kerns
        if _inputs.ndim != 4:
            raise TypeError('make_node requires 4D tensor of inputs')
        if _kerns.ndim != 4:
            raise TypeError('make_node requires 4D tensor of kernels')
        if _inputs.type.dtype != _kerns.type.dtype:
            raise NotImplementedError("The image and the kernel must have the same type."
                            "inputs(%s), kerns(%s)"%(_inputs.dtype, _kerns.dtype))
        output = CudaNdarrayType(dtype=_inputs.type.dtype,
                                 broadcastable=[_inputs.broadcastable[0],
                                                _kerns.broadcastable[0], 
                                                False, False])()

        return Apply(self, [_inputs, _kerns], [output])

    def c_headers(self):
        return ['<numpy/noprefix.h>', '<cuda.h>', '<cuda_runtime.h>','<cufft.h>','<cuComplex.h>','<stdlib.h>','<stdint.h>','<unistd.h>','<sys/time.h>','<time.h>']

    def c_libraries(self):
        return ['cufft']

    def c_compile_args(self):
        if self.debug:
            return ['-DDEBUG']
        else:
            return []

    #def c_code_cache_version(self):
    #    return (4)

    def c_support_code(self):
        check = ""
        if self.check:
            check = "#define CHECK"
        return """
// TODO: Why are these necessary? What's wrong with stdint.h??
typedef unsigned int uint32;
typedef int int32;


int verbose = 0;
%(check)s

cufftHandle fwd_plan;
cufftHandle inv_plan;
int32 old_padded_dimensions[2]={-1,-1};
uint32 old_num_padded = 0;
uint32 old_inv_plan_size = 0;

int check_success(char * str){
    cudaThreadSynchronize();
    cudaError_t sts = cudaGetLastError();
    if (cudaSuccess == sts){
        if (verbose) fprintf(stderr,"INFO: GpuFFTConvOp %%s succeded\\n", str);
        return true;
    }else{
        PyErr_Format(PyExc_TypeError, "INFO: GpuFFTConvOp %%s failed (%%s). Run with this op in debug mode.", str, cudaGetErrorString(sts));
        if (verbose) fprintf(stderr,"INFO: GpuFFTConvOp %%s failed(%%s). Run with this op in debug mode.\\n", str,
                            cudaGetErrorString(sts));
        return false;
    }
}

uint32 next_power_of_two(uint32 i) {
    // from Sean Anderson's bit twiddling hacks at http://www-graphics.stanford.edu/~seander/bithacks.html
    i--;
    i |= i >> 1;
    i |= i >> 2;
    i |= i >> 4;
    i |= i >> 8;
    i |= i >> 16;
    i++;
    return i;
}

// TODO: is it possible to optimize this?
__global__ void pad_images_and_kernels(float *images,
                                       float *kernels,
                                       float *dest,
                                       uint32 total_images,
                                       uint32 image_rows,
                                       uint32 image_cols,
                                       uint32 kernel_rows,
                                       uint32 kernel_cols,
                                       uint32 padded_rows,
                                       uint32 padded_cols) {
    uint32 ik_offset = blockIdx.x;
    
    uint32 source_rows;
    uint32 source_cols;
    float *src;
    uint32 src_offset;
    if(ik_offset < total_images) {
        source_rows = image_rows;
        source_cols = image_cols;
        src = images;
        src_offset = ik_offset;
    } else {
        source_rows = kernel_rows;
        source_cols = kernel_cols;
        src = kernels;
        src_offset = ik_offset - total_images;
    }
    uint32 col_index = threadIdx.y;

    for(uint32 row_index = threadIdx.x; row_index<padded_rows; row_index+=blockDim.x){
        float out;
        if(row_index >= source_rows || col_index >= source_cols) {
            out = 0.0f;
        } else {
            out = src[src_offset * source_rows * source_cols + row_index * source_cols + col_index];
        }
    
        dest[ik_offset * padded_rows * padded_cols + row_index * padded_cols + col_index] = out;
    }
}

// TODO: is it possible to optimize this? reduce access to global mem by loading into shared mem, etc.
__global__ void elementwise_image_kernel_multiply(cufftComplex *transformed,
                                                  cufftComplex *multiplied,
                                                  uint32 batch_size,
                                                  uint32 num_kernels,
                                                  uint32 num_images,
                                                  uint32 element_length) {
    uint32 batch_kernel_index = blockIdx.x;
    uint32 batch_index = batch_kernel_index / num_kernels;
    uint32 kernel_index = batch_kernel_index %% num_kernels;
    uint32 image_index = blockIdx.y;
    uint32 element_index = threadIdx.x;
    for(uint32 element_index = threadIdx.x;element_index<element_length;element_index+=blockDim.x){
        cufftComplex *image_src = transformed
                                + batch_index * num_images * element_length
                                + image_index * element_length
                                + element_index;

        cufftComplex *transformed_kernels = transformed + batch_size * num_images * element_length;
        cufftComplex *kernel_src = transformed_kernels
                                 + kernel_index * num_images * element_length
                                 + image_index * element_length
                                 + element_index;

        cufftComplex *dest = multiplied
                           + batch_index * num_kernels * num_images * element_length
                           + kernel_index * num_images * element_length
                           + image_index * element_length
                           + element_index;

    *dest = cuCmulf(*image_src, *kernel_src);
    }
}

// TODO: is it possible to optimize this?
__global__ void add_across_images_and_normalize(float *inverse_transformed,
                                                float *added,
                                                uint32 num_images,
                                                uint32 batch_size,
                                                uint32 num_kernels,
                                                uint32 padded_rows,
                                                uint32 padded_cols,
                                                float normalization_factor) {
    uint32 col = threadIdx.y;
    uint32 batch_index = blockIdx.x;
    uint32 kernel_index = blockIdx.y;
    for(uint32 row = threadIdx.x;row<padded_rows;row+=blockDim.x){
        float sum = 0.0f;
        for(uint32 image_index = 0; image_index < num_images; image_index++) {
          float *image = inverse_transformed + (batch_index * num_kernels * num_images + kernel_index * num_images + image_index) * padded_rows * padded_cols;
          float *image_element = image + row * padded_cols + col;
          sum += *image_element;
        }
        float *added_destination = added + batch_index * num_kernels * padded_rows * padded_cols + kernel_index * padded_rows * padded_cols + row * padded_cols + col;
        *added_destination = sum / normalization_factor;
    }
}
 """ %locals()

    def c_code(self, node, name, (img, kern), (z, ), sub):
        if node.inputs[0].type.dtype != node.inputs[1].type.dtype:
            raise NotImplementedError()
        assert node.inputs[0].type.dtype == node.inputs[1].type.dtype
        d=locals()
        d.update(sub)
        more_memory = int(self.more_memory)
        return """
    const int shared_avail = SHARED_SIZE-150;//144 is the biggest static shared size used with compiling this file.
    CudaNdarray *img = %(img)s;
    CudaNdarray * kern = %(kern)s;
#ifdef DEBUG
printf("z=%%p\\n",%(z)s);//Why in mode FAST_RUN_NOGC, we don't have it already allocated?
#endif
    CudaNdarray * out = %(z)s;

    if (img->nd != 4){
        PyErr_SetString(PyExc_ValueError, "GpuFFTConvOp required img of 4D");
        return NULL;
    }
    if (kern->nd != 4){
        PyErr_SetString(PyExc_ValueError, "GpuFFTConvOp required kern of 4D");
        return NULL;
    }
    if(!CudaNdarray_is_c_contiguous(img)){
        PyErr_SetString(PyExc_ValueError, "GpuFFTConvOp required c contiguous images");
        return NULL;
    }
    if(!CudaNdarray_is_c_contiguous(kern)){
        PyErr_SetString(PyExc_ValueError, "GpuFFTConvOp required c contiguous kernel");
        return NULL;
    }

    int out_dim[4];
    out_dim[0] = CudaNdarray_HOST_DIMS(img)[0];
    out_dim[1] = CudaNdarray_HOST_DIMS(kern)[0];

    int logical_rows, logical_cols;
    if (false)//mode == ConvMode_VALID)
    {
        logical_rows = CudaNdarray_HOST_DIMS(img)[2] - CudaNdarray_HOST_DIMS(kern)[2] + 1;
        logical_cols = CudaNdarray_HOST_DIMS(img)[3] - CudaNdarray_HOST_DIMS(kern)[3] + 1;
    }
    else
    {
        logical_rows = CudaNdarray_HOST_DIMS(img)[2] + CudaNdarray_HOST_DIMS(kern)[2] - 1;
        logical_cols = CudaNdarray_HOST_DIMS(img)[3] + CudaNdarray_HOST_DIMS(kern)[3] - 1;
    }
    out_dim[2] = logical_rows;
    out_dim[3] = logical_cols;
    
    if(!(out && out->nd==4 && CudaNdarray_is_c_contiguous(out) 
	 && CudaNdarray_HOST_DIMS(out)[0]==out_dim[0]
	 && CudaNdarray_HOST_DIMS(out)[1]==out_dim[1]
	 && CudaNdarray_HOST_DIMS(out)[2]==out_dim[2]
	 && CudaNdarray_HOST_DIMS(out)[3]==out_dim[3])){
      if(out && verbose)
          printf("Will allocate a new output memory %%p %%d %%d %%d %%d %%d %%d \\n",
             0, out->nd==4, CudaNdarray_is_c_contiguous(out),
	     CudaNdarray_HOST_DIMS(out)[0]==out_dim[0],
     	     CudaNdarray_HOST_DIMS(out)[1]==out_dim[1],
	     CudaNdarray_HOST_DIMS(out)[2]==out_dim[2],
	     CudaNdarray_HOST_DIMS(out)[3]==out_dim[3]);
      else if(verbose)
          printf("Will allocate a new output memory\\n");

      out = (CudaNdarray*)CudaNdarray_NewDims(4,out_dim);
    }

    if (!out)
    {
        PyErr_SetString(PyExc_ValueError, "not able to create a new output image");
        return NULL;
    }
 

    const int nstack=CudaNdarray_HOST_DIMS(kern)[1];
    const int nbatch=CudaNdarray_HOST_DIMS(img)[0];
    const int nkern=CudaNdarray_HOST_DIMS(kern)[0];
    const int img_wid=CudaNdarray_HOST_DIMS(img)[3];
    const int img_len=CudaNdarray_HOST_DIMS(img)[2];
    const int kern_wid=CudaNdarray_HOST_DIMS(kern)[3];
    const int kern_len=CudaNdarray_HOST_DIMS(kern)[2];
    const int out_wid=CudaNdarray_HOST_DIMS(out)[3];
    const int out_len=CudaNdarray_HOST_DIMS(out)[2];

    const int img_stride_col= CudaNdarray_HOST_STRIDES(img)[3];
    const int img_stride_row=CudaNdarray_HOST_STRIDES(img)[2];
    const int img_stride_stack=CudaNdarray_HOST_STRIDES(img)[1];
    const int img_stride_batch=CudaNdarray_HOST_STRIDES(img)[0];
    const int kern_stride_col= CudaNdarray_HOST_STRIDES(kern)[3];
    const int kern_stride_row=CudaNdarray_HOST_STRIDES(kern)[2];
    const int kern_stride_stack= CudaNdarray_HOST_STRIDES(kern)[1];
    const int kern_stride_nkern=CudaNdarray_HOST_STRIDES(kern)[0];

    const int img_size=img_len*img_wid;
    const int kern_size=kern_len*kern_wid;
    const int out_size=out_len*out_wid;
    const int img_size_byte = img_size*sizeof(float);
    const int kern_size_byte = kern_size*sizeof(float);
    //padded image sizes
    const int img_wid_padded=img_wid+2*kern_wid-2;
    const int img_len_padded=img_len+2*kern_len-2;
    const int img_size_padded=img_len_padded * img_wid_padded;
    const int img_size_padded_byte = img_size_padded*sizeof(float);


    uint32 padded_rows = next_power_of_two(out_len);
    uint32 padded_cols = next_power_of_two(out_wid);
    uint32 num_padded = nbatch * nstack + nkern * nstack; // total images + total kernels
    int32 transformed_cols = padded_cols / 2 + 1; // only non-redundant complex coefficients are calculated




//SHOULD BE DONE ONLY ONCE
    // assume we can pay the planning price just once and amortize it away, so do the planning up front
    int32 padded_dimensions[2] = {padded_rows, padded_cols};
    if(!(fwd_plan &&
         old_padded_dimensions[0] == padded_dimensions[0] &&
         old_padded_dimensions[1] == padded_dimensions[1] &&
         old_num_padded == num_padded)){
        if(verbose)
            printf("create new fwd_plan %%p old_padded_dimensions=(%%d,%%d) padded_dimensions=(%%d,%%d) old_num_padded=%%d num_padded=%%d\\n", fwd_plan,
            old_padded_dimensions[0],old_padded_dimensions[1],padded_dimensions[0],padded_dimensions[1],old_num_padded,num_padded);
        cufftResult plan_result = cufftPlanMany(&fwd_plan, // plan
                                            2, // rank
                                            padded_dimensions, // dimensions
                                            NULL, 1, 0, NULL, 1, 0, // boilerplate for contiguous access (non-contiguous access not supported now)
                                            CUFFT_R2C, // fwd transform, real to complex
                                            num_padded // fft batch size
                                           );
        cufftSetCompatibilityMode(fwd_plan, CUFFT_COMPATIBILITY_NATIVE); // performance only
    }

    if(!(inv_plan &&
         old_padded_dimensions[0] == padded_dimensions[0] &&
         old_padded_dimensions[1] == padded_dimensions[1] &&
         old_inv_plan_size == (nbatch * nkern * nstack))){
        if(verbose)
            printf("create new inv_plan %%p old_padded_dimensions=(%%d,%%d) padded_dimensions=(%%d,%%d) old_inv_plan_size=%%d inv_plan_size=%%d\\n", inv_plan,
            old_padded_dimensions[0],old_padded_dimensions[1],padded_dimensions[0],padded_dimensions[1],old_inv_plan_size,nbatch * nkern * nstack);
        cufftPlanMany(&inv_plan, // plan
                  2, // rank
                  padded_dimensions, // dimensions
                  NULL, 1, 0, NULL, 1, 0, // boilerplate for contiguous access (non-contiguous access not supported now)
                  CUFFT_C2R, // inv transform, complex to real
                  nbatch * nkern * nstack // ifft batch size
                 );
         // CUFFT_COMPATIBILITY_NATIVE needed to prevent extra padding, 
         // so output is compact and nicely accessible via c pointer arithmetic
         cufftSetCompatibilityMode(inv_plan, CUFFT_COMPATIBILITY_NATIVE);
     }
     old_padded_dimensions[0] = padded_dimensions[0];
     old_padded_dimensions[1] = padded_dimensions[1];
     old_inv_plan_size = (nbatch * nkern * nstack);
     old_padded_dimensions[0] = padded_dimensions[0];
     old_padded_dimensions[1] = padded_dimensions[1];
     old_num_padded = num_padded;

//SHOULD BE DONE AT EACH CALL
#if DEBUG

printf("GpuFFTConvOp before init inv[nbatch%%d][nkern%%d][nstack%%d][padded_rows%%d][padded_cols%%d]\\n",nbatch,nkern,nstack,padded_rows,padded_cols);
    float inv[nbatch][nkern][nstack][padded_rows][padded_cols];
#endif

    dim3 adding_grid(nbatch, nkern);
    dim3 adding_threads(padded_rows, padded_cols);
    dim3 dim_grid(nbatch * nkern, nstack);
    dim3 dim_thread(padded_rows * transformed_cols);
    dim3 padding_threads(padded_rows, padded_cols);

    //create 4 temporary space in one cudaMalloc call to make it faster.
    int fft_input_size = sizeof(float) * num_padded * padded_rows * padded_cols;
    int transformed_size = sizeof(cufftComplex) * num_padded * padded_rows * transformed_cols;
    int multiplied_size = sizeof(cufftComplex) * nbatch * nkern * nstack * padded_rows * transformed_cols;
    int inverse_transformed_size = sizeof(float) * nbatch * nkern * nstack * padded_rows * padded_cols;

    void * device_mem = NULL;
    float *fft_input = NULL;
    cufftComplex *transformed = NULL;
    cufftComplex *multiplied = NULL;
    float *inverse_transformed = NULL ;
    cudaMalloc(&device_mem, fft_input_size+transformed_size+multiplied_size+inverse_transformed_size);
#ifdef CHECK
if(!check_success("cudaMalloc(device_mem,...)")){
        Py_XDECREF(out);
        out = NULL;
        %(fail)s;
}
#endif

    fft_input = ((float*)device_mem);
    transformed = ((cufftComplex*)(fft_input+fft_input_size/sizeof(float)));
    multiplied = ((cufftComplex*)(transformed+transformed_size/sizeof(cufftComplex)));
    inverse_transformed = ((float*)(multiplied+multiplied_size/sizeof(cufftComplex)));















    // rearrange images and kernels to their new padded size, all contiguous
    // to each other, since that is what the batched fft requires right now
    assert(padded_cols<=512);
    while(padding_threads.x*padding_threads.y>512)padding_threads.x--;
    pad_images_and_kernels<<<num_padded, padding_threads>>>(img->devdata,
                                                            kern->devdata,
                                                            fft_input,
                                                            nbatch * nstack, // number of images in the fft
                                                            img_len,
                                                            img_wid,
                                                            kern_len,
                                                            kern_wid,
                                                            padded_rows,
                                                            padded_cols);

#ifdef CHECK
if(!check_success("pad_images_and_kernels")){
        fprintf(stderr,"num_padded=(%%d), padding_threads=(%%d,%%d)\\n",num_padded,
               padding_threads.x, padding_threads.y);
        Py_XDECREF(out);
        out = NULL;
        %(fail)s;
}
#endif

#if DEBUG
    fprintf(stderr, "PADDED\\n");
    float pad[num_padded][padded_rows][padded_cols];
    cudaMemcpy(pad, fft_input, sizeof(float) * num_padded * padded_rows * padded_cols, cudaMemcpyDeviceToHost);
    for(uint32 n = 0; n < num_padded; n++) {
        for(uint32 r = 0; r < padded_rows; r++) {
            for(uint32 c = 0; c < padded_cols; c++) {
                fprintf(stderr,
                        "%%.0f ", pad[n][r][c]);
            }
            fprintf(stderr, "\\n");
        }
        fprintf(stderr, "\\n");
    }
#endif
    
    // perform forward fft
    cufftExecR2C(fwd_plan, fft_input, transformed);
#ifdef CHECK
if(!check_success("cufftExecR2C")){
        Py_XDECREF(out);
        out = NULL;
        %(fail)s;
}
#endif

    // do elemwise multiplication
    if(dim_thread.x>512)dim_thread.x=512;
    elementwise_image_kernel_multiply<<<dim_grid, dim_thread>>>(
            transformed,
            multiplied,
            nbatch,
            nkern,
            nstack,
            padded_rows * transformed_cols);
#ifdef CHECK
if(!check_success("elementwise_image_kernel_multiply")){
        printf("dim_grid=(%%d,%%d) nb_threads=%%d\\n",
               dim_grid.x,dim_grid.y,
               padded_rows * transformed_cols);
        Py_XDECREF(out);
        out = NULL;
        %(fail)s;
}
#endif

    // do inverse fft
    cufftExecC2R(inv_plan, multiplied, inverse_transformed);
#ifdef CHECK
if(!check_success("cufftExecC2R")){
        Py_XDECREF(out);
        out = NULL;
        %(fail)s;
}
#endif

#if DEBUG
    fprintf(stderr, "INVERSE_TRANSFORMED\\n");
    //float inv[nbatch][nkern][nstack][padded_rows][padded_cols];
    cudaMemcpy(inv, inverse_transformed, sizeof(float) * nbatch * nkern * nstack * padded_rows * padded_cols, cudaMemcpyDeviceToHost);
    for(uint32 b = 0; b < nbatch; b++) {
        for(uint32 k = 0; k < nkern; k++) {
            for(uint32 i = 0; i < nstack; i++) {
                fprintf(stderr, "<trans b %%i, k %%i, i %%i>\\n", b, k, i);
                for(uint32 r = 0; r < padded_rows; r++) {
                    for(uint32 c = 0; c < padded_cols; c++) {
                        fprintf(stderr,
                                "%%.0f ", inv[b][k][i][r][c]);
                    }
                    fprintf(stderr, "\\n");
                }
                fprintf(stderr, "\\n");
            }
        }
    }
#endif


    // sum across images and scale the results appropriately (cufft does non-normalized transforms)
    assert(adding_threads.y<512);
    while(adding_threads.x*adding_threads.y>512)adding_threads.x--;
    add_across_images_and_normalize<<<adding_grid, adding_threads>>>(
        inverse_transformed,
        out->devdata,
        nstack,
        nbatch,
        nkern,
        padded_rows,
        padded_cols,
        padded_rows * padded_cols); // normalization factor
#ifdef CHECK
if(!check_success("add_across_images_and_normalize")){
        printf("dim_grid=(%%d,%%d) nb_threads=(%%d,%%d)\\n",
               adding_grid.x,adding_grid.y,
               adding_threads.x,adding_threads.y);
        Py_XDECREF(out);
        out = NULL;
        %(fail)s;
}
#endif
    if(!%(more_memory)s){
        cufftDestroy(fwd_plan);
        fwd_plan = NULL;
    #ifdef CHECK
    if(!check_success("cufftDestroy(fwd_plan)")){
        Py_XDECREF(out);
        out = NULL;
        %(fail)s;
    }
    #endif
    }
    if(!%(more_memory)s){
        cufftDestroy(inv_plan);
        inv_plan = NULL;
    #ifdef CHECK
    if(!check_success("cufftDestroy(inv_plan)")){
        Py_XDECREF(out);
        out = NULL;
        %(fail)s;
    }
    #endif
    }

    cudaFree(device_mem);
#ifdef CHECK
if(!check_success("cudaFree(device_mem)")){
        Py_XDECREF(out);
        out = NULL;
        %(fail)s;
}
#endif

//needed in to make the cudaThreadSynchronize and check if any of the previous
//call failed.
#ifndef CHECK
    if(!check_success("globalgpu kernell calls")){
        Py_XDECREF(out);
        out = NULL;
        %(fail)s;
    }
#endif

    %(z)s=out;
 """%locals()

#gpu_fft_conv = GpuFFTConvOp()
from theano.sandbox.cuda.opt import register_opt
from theano.sandbox.cuda.blas import GpuConv
from theano.gof import local_optimizer

@register_opt()
@local_optimizer([GpuConv])
def local_gpu_fft_conv(node):
    """
    gpu_conv -> gpu_fft_conv_op

    """
    if (isinstance(node.op, GpuConv) and 
        node.op.border_mode=='full' and 
        node.op.subsample==(1,1)):
        img, kern = node.inputs
        gpu_fft_conv = GpuFFTConvOp(node.op.border_mode, check=node.op.verbose)
        return [gpu_fft_conv(img,kern)]
