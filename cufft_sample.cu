// build with nvcc cufft_sample.cu -lcufft

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>

// TODO: Why are these necessary? What's wrong with stdint.h??
typedef unsigned int uint32;
typedef int int32;


// TODO: optimize me
__global__ void pad_images_and_kernels(cufftReal *images,
                                       cufftReal *kernels,
                                       cufftReal *dest,
                                       uint32 num_images,
                                       uint32 image_rows,
                                       uint32 image_cols,
                                       uint32 kernel_rows,
                                       uint32 kernel_cols,
                                       uint32 padded_rows,
                                       uint32 padded_cols) {
    uint32 row_index = threadIdx.x;
    uint32 col_index = threadIdx.y;
    uint32 ik_offset = blockIdx.x;

    uint32 source_rows;
    uint32 source_cols;
    cufftReal *src;
    uint32 src_offset;
    if(ik_offset < num_images) {
        source_rows = image_rows;
        source_cols = image_cols;
        src = images;
        src_offset = ik_offset;
    } else {
        source_rows = kernel_rows;
        source_cols = kernel_cols;
        src = kernels;
        src_offset = ik_offset - num_images;
    }

    cufftReal out;
    if(row_index >= source_rows || col_index >= source_cols) {
        out = 0.0f;
    } else {
        out = src[src_offset * source_rows * source_cols + row_index * source_cols + col_index];
    }
    
    dest[ik_offset * padded_rows * padded_cols + row_index * padded_cols + col_index] = out;
}

// TODO: optimize me more
// TODO: what happens when element_length is bigger than the allowed num threads?
__global__ void elementwise_image_kernel_multiply(cufftComplex *transformed,
                                                  cufftComplex *multiplied,
                                                  uint32 num_images,
                                                  uint32 num_kernels,
                                                  uint32 element_length) {
    uint32 image_index = blockIdx.x;
    uint32 kernel_index = blockIdx.y;
    uint32 element_index = threadIdx.x;
    cufftComplex *image_src = transformed + image_index * element_length;
    cufftComplex *kernel_src = transformed + (num_images + kernel_index) * element_length;
    cufftComplex *dest = multiplied + (image_index * num_kernels + kernel_index) * element_length;

    dest[element_index] = cuCmulf(image_src[element_index], kernel_src[element_index]);
}

// TODO: optimize me
__global__ void elementwise_vector_scalar_multiply_inplace(cufftReal *vector, float scalar, uint32 len) {
    uint32 index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len) {
        vector[index] = vector[index] * scalar;
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

uint32 max_threads_per_block() {
    uint32 device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int max_threads_per_block = (deviceProp.major >= 2 ? 512 : 256);
    return max_threads_per_block;
}

// NB: This will die at runtime sometimes in cuda 3.1 due to
// data alignment issues -- see http://forums.nvidia.com/index.php?showtopic=176207&mode=linearplus
// TODO: Don't use it w/ 3.1. (How to detect?)
int main(int argc, char *argv[])
{
    // general cuda setup and prep
    int threads_per_block = max_threads_per_block();
    
#if RUN_SPEED_TESTS
    uint32 num_images = 50;
#else
    uint32 num_images = 2;
#endif
    uint32 image_rows = 8;
    uint32 image_cols = 8;

#if RUN_SPEED_TESTS
    uint32 num_kernels = 20;
#else
    uint32 num_kernels = 2;
#endif
    uint32 kernel_rows = 5;
    uint32 kernel_cols = 5;

    uint32 num_padded = num_images + num_kernels;
    uint32 padded_rows = next_power_of_two(image_rows + kernel_rows - 1);
    uint32 padded_cols = next_power_of_two(image_cols + kernel_cols - 1);

    // set up images
    cufftReal images[num_images][image_rows][image_cols];
    for(uint32 b = 0; b < num_images; b++) {
        for(uint32 r = 0; r < image_rows; r++) {
            for(uint32 c = 0; c < image_cols; c++) {
                images[b][r][c] = (float)b + 1;
            }
        }
    }

    cufftReal kernels[num_kernels][kernel_rows][kernel_cols];
    for(uint32 b = 0; b < num_kernels; b++) {
        for(uint32 r = 0; r < kernel_rows; r++) {
            for(uint32 c = 0; c < kernel_cols; c++) {
                kernels[b][r][c] = (float)b + 1;
            }
        }
    }

    // copy images and kernels to device
    // do this first, and simply (before padding etc., to mimic what theano
    // would have to do -- presumably, the data is already on the gpu)
    cufftReal *inbound_images;
    cudaMalloc((void**)&inbound_images, sizeof(cufftReal) * num_images * image_rows * image_cols);
    cudaMemcpy(inbound_images, images, sizeof(cufftReal) * num_images * image_rows * image_cols, cudaMemcpyHostToDevice);

    cufftReal *inbound_kernels;
    cudaMalloc((void**)&inbound_kernels, sizeof(cufftReal) * num_kernels * kernel_rows * kernel_cols);
    cudaMemcpy(inbound_kernels, kernels, sizeof(cufftReal) * num_kernels * kernel_rows * kernel_cols, cudaMemcpyHostToDevice);


    // assume we can pay the planning price just once and amortize it away, so do the planning up front
    int32 padded_dimensions[2] = {padded_rows, padded_cols};

    cufftHandle fwd_plan;
    cufftResult plan_result = cufftPlanMany(&fwd_plan, // plan
                                            2, // rank
                                            padded_dimensions, // dimensions
                                            NULL, 1, 0, NULL, 1, 0, // boilerplate for contiguous access (non-contiguous access not supported now)
                                            CUFFT_R2C, // type
                                            num_padded // batch size
                                           );
    cufftSetCompatibilityMode(fwd_plan, CUFFT_COMPATIBILITY_NATIVE);
    
    cufftHandle inv_plan;
    cufftPlanMany(&inv_plan, // plan
                  2, // rank
                  padded_dimensions, // dimensions
                  NULL, 1, 0, NULL, 1, 0, // boilerplate for contiguous access (non-contiguous access not supported now)
                  CUFFT_C2R, // type
                  num_images * num_kernels // batch size
                 );
     cufftSetCompatibilityMode(inv_plan, CUFFT_COMPATIBILITY_NATIVE); // needed to prevent extra padding, so output looks natural

    // ok, the data is on the gpu; the plans are made; start the timer
    uint32 num_iterations = 100000;
    time_t start, end;

    start = time(NULL);


#if RUN_SPEED_TESTS
    for(uint32 iteration = 0; iteration < num_iterations; iteration++) {
#endif

    // rearrange images and kernels to their new padded size, all contiguous
    // to each other, since that is what the batched fft requires right now
    
    cufftReal *fft_input;
    cudaMalloc((void**)&fft_input, sizeof(cufftReal) * num_padded * padded_rows * padded_cols);
    
    dim3 padding_threads(padded_rows, padded_cols);
    pad_images_and_kernels<<<num_padded, padding_threads>>>(inbound_images,
                                                            inbound_kernels,
                                                            fft_input,
                                                            num_images,
                                                            image_rows,
                                                            image_cols,
                                                            kernel_rows,
                                                            kernel_cols,
                                                            padded_rows,
                                                            padded_cols);


    /****************** 11s to here *************/
    // perform forward fft
    uint32 transformed_cols = padded_cols / 2 + 1; // only non-redundant complex coefficients are calculated

    cufftComplex *transformed;
    cudaMalloc((void**)&transformed, sizeof(cufftComplex) * num_padded * padded_rows * transformed_cols);

    cufftResult fwd_result = cufftExecR2C(fwd_plan, fft_input, transformed);
    if(fwd_result != CUFFT_SUCCESS) {
        fprintf(stderr, "fwd fft failed: %i", fwd_result);
    }

    /****************** 11s to here, too! *************/

    // do elemwise multiplication
    cufftComplex *multiplied;
    // this memory will be re-used when the C2R transform is done in place, so make sure there's enough space
    uint32 multiplied_size = sizeof(cufftComplex) * num_images * num_kernels * padded_rows * transformed_cols;
    uint32 inverse_transformed_size = sizeof(cufftReal) * num_images * num_kernels * padded_rows * padded_cols;
    cudaMalloc((void**)&multiplied, max(multiplied_size, inverse_transformed_size));

    dim3 dim_grid(num_images, num_kernels);
    elementwise_image_kernel_multiply<<<dim_grid, padded_rows * transformed_cols>>>(transformed, multiplied, num_images, num_kernels, padded_rows * transformed_cols);

    /****************** 13s to here *************/

    // do inverse fft
    cufftReal *inverse_transformed;
    inverse_transformed = (cufftReal *)multiplied;

    cufftExecC2R(inv_plan, multiplied, inverse_transformed);

    // scale the results appropriately (cufft does non-normalized transforms)
    int blocks_per_grid = (num_images * num_kernels * padded_rows * padded_cols + threads_per_block - 1) / threads_per_block;
    elementwise_vector_scalar_multiply_inplace<<<blocks_per_grid, threads_per_block>>>(inverse_transformed,
                                                                                       1.0f / (padded_rows * padded_cols),
                                                                                       num_images * num_kernels * padded_rows * padded_cols);

    /****************** 15s to here *************/
    cudaFree(fft_input);
#if RUN_SPEED_TESTS
    cudaFree(transformed);
    cudaFree(multiplied);
#endif

#if RUN_SPEED_TESTS
    } // end timing-iteration for loop
#endif

    // all the calculations are (basically) done, at least for full mode
    // stop the timer
    end = time(NULL);

    fprintf(stderr, 
            "%i %ix%i images, %i %ix%i kernels: %.0fsec elapsed for %i iterations (%f sec/iter)\n",
            num_images,
            image_rows,
            image_cols,
            num_kernels,
            kernel_rows,
            kernel_cols,
            difftime(end, start),
            num_iterations,
            difftime(end, start) / (float)num_iterations);

#if RUN_SPEED_TESTS
#else

    // TODO: Set strides appropriately or do memcpys to get rid of unneeded padding
    float results[num_images][num_kernels][padded_rows][padded_cols];
    // copy results back to host
    cudaMemcpy(results, inverse_transformed, sizeof(cufftReal) * num_images * num_kernels * padded_rows * padded_cols, cudaMemcpyDeviceToHost);


    fprintf(stderr, "OUTBOUND\n");
    for(uint32 image_index = 0; image_index < num_images; image_index++) {
        for(uint32 kernel_index = 0; kernel_index < num_kernels; kernel_index++) {
            for(uint32 r = 0; r < padded_rows; r++) {
                for(uint32 c = 0; c < padded_cols; c++) {
                    fprintf(stderr, "%.0f ", results[image_index][kernel_index][r][c]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
    }
#endif

    // TODO: Other cleanup -- memory freeing, etc.
    cufftDestroy(fwd_plan); // TODO: reuse fft plans
    cufftDestroy(inv_plan);
}
