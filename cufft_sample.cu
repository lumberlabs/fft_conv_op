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
// optimizations include preloading (say) kernels and iterating over images
// etc. etc.
__global__ void elementwise_matrix_matrix_multiply(cufftComplex *src1, cufftComplex *src2, cufftComplex *dest, uint32 len) {
    uint32 index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len) {
        dest[index] = cuCmulf(src1[index], src2[index]);
    }
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
    
    uint32 num_images = 1;
    uint32 image_rows = 28;
    uint32 image_cols = 28;

    uint32 num_kernels = 50;
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
                images[b][r][c] = (float)b;
            }
        }
    }

    cufftReal kernels[num_kernels][kernel_rows][kernel_cols];
    for(uint32 b = 0; b < num_kernels; b++) {
        for(uint32 r = 0; r < kernel_rows; r++) {
            for(uint32 c = 0; c < kernel_cols; c++) {
                kernels[b][r][c] = (float)b;
            }
        }
    }

/*
    fprintf(stderr, "INBOUND IMAGES\n");
    for(uint32 image_index = 0; image_index < num_images; image_index++) {
        for(uint32 r = 0; r < image_rows; r++) {
            for(uint32 c = 0; c < image_cols; c++) {
                fprintf(stderr, "%.0f ", images[image_index][r][c]);
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }


    fprintf(stderr, "INBOUND KERNELS\n");
    for(uint32 kernel_index = 0; kernel_index < num_kernels; kernel_index++) {
        for(uint32 r = 0; r < kernel_rows; r++) {
            for(uint32 c = 0; c < kernel_cols; c++) {
                fprintf(stderr, "%.0f ", kernels[kernel_index][r][c]);
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }
*/

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
    
    cufftHandle inv_plan;
    cufftPlanMany(&inv_plan, // plan
                  2, // rank
                  padded_dimensions, // dimensions
                  NULL, 1, 0, NULL, 1, 0, // boilerplate for contiguous access (non-contiguous access not supported now)
                  CUFFT_C2R, // type
                  num_images * num_kernels // batch size
                 );


    // ok, the data is on the gpu; the plans are made; start the timer
    uint32 num_iterations = 10000;
    time_t start, end;

    start = time(NULL);

    
    for(uint32 iteration = 0; iteration < num_iterations; iteration++) {
    ////////////////////////////////////////////////////////////////////

    // rearrange images and kernels to their new padded size, all contiguous
    // to each other, since that is what the batched fft requires right now
    
    // TODO: Would writing a custom kernel to do this memory shuffling be faster?
    cufftReal *fft_input;
    cudaMalloc((void**)&fft_input, sizeof(cufftReal) * num_padded * padded_rows * padded_cols);
    cudaMemset(fft_input, 0, sizeof(cufftReal) * num_padded * padded_rows * padded_cols);

    for(int b = 0; b < num_images; b++) {
        for(int r = 0; r < image_rows; r++) {
            cudaMemcpy(fft_input + b * padded_rows * padded_cols + r * padded_cols,
                       inbound_images + b * image_rows * image_cols + r * image_cols,
                       sizeof(cufftReal) * image_cols,
                       cudaMemcpyDeviceToDevice);
        }
    }

    for(int b = 0; b < num_kernels; b++) {
        for(int r = 0; r < kernel_rows; r++) {
            cudaMemcpy(fft_input + (b + num_images) * padded_rows * padded_cols + r * padded_cols,
                       inbound_kernels + b * kernel_rows * kernel_cols + r * kernel_cols,
                       sizeof(cufftReal) * kernel_cols,
                       cudaMemcpyDeviceToDevice);
        }
    }
/*
    fprintf(stderr, "FFT INPUT\n");
    float fi[num_padded][padded_rows][padded_cols];
    cudaMemcpy(fi, fft_input, sizeof(cufftReal) * num_padded * padded_rows * padded_cols, cudaMemcpyDeviceToHost);
    for(uint32 padded_index = 0; padded_index < num_padded; padded_index++) {
        for(uint32 r = 0; r < padded_rows; r++) {
            for(uint32 c = 0; c < padded_cols; c++) {
                fprintf(stderr, "%.0f ", fi[padded_index][r][c]);
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }
*/
    
    // perform forward fft
    uint32 transformed_cols = padded_cols / 2 + 1; // only non-redundant complex coefficients are calculated

    cufftComplex *transformed;
    cudaMalloc((void**)&transformed, sizeof(cufftComplex) * num_padded * padded_rows * transformed_cols);

    cufftResult fwd_result = cufftExecR2C(fwd_plan, fft_input, transformed);
    if(fwd_result != CUFFT_SUCCESS) {
        fprintf(stderr, "fwd fft failed: %i", fwd_result);
    }

/*
    fprintf(stderr, "FFT OUTPUT\n");
    cuComplex fo[num_padded][padded_rows][transformed_cols];
    cudaMemcpy(fo, transformed, sizeof(cufftComplex) * num_padded * padded_rows * transformed_cols, cudaMemcpyDeviceToHost);
    for(uint32 padded_index = 0; padded_index < num_padded; padded_index++) {
        for(uint32 r = 0; r < padded_rows; r++) {
            for(uint32 c = 0; c < transformed_cols; c++) {
                fprintf(stderr, "%.0f,%.0f ", cuCrealf(fo[padded_index][r][c]), cuCimagf(fo[padded_index][r][c]));
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }
*/

    // TODO: memory cleanup as appropriate
    // cudaFree(images);

    // do elemwise multiplication
    cufftComplex *multiplied;
    cudaMalloc((void**)&multiplied, sizeof(cufftComplex) * num_images * num_kernels * padded_rows * transformed_cols);
    cudaMemset(multiplied, 0xFF, sizeof(cufftComplex) * num_images * num_kernels * padded_rows * transformed_cols); // TODO: DEBUGGING ONLY!

    int blocks_per_grid = (padded_rows * transformed_cols + threads_per_block - 1) / threads_per_block;
    for(uint32 image_index = 0; image_index < num_images; image_index++) {
        for(uint32 kernel_index = 0; kernel_index < num_kernels; kernel_index++) {
            elementwise_matrix_matrix_multiply<<<blocks_per_grid, threads_per_block>>>(transformed + image_index * padded_rows * transformed_cols,
                                                                                       transformed + (num_images + kernel_index) * padded_rows * transformed_cols,
                                                                                       multiplied + (image_index * num_kernels + kernel_index) * padded_rows * transformed_cols,
                                                                                       padded_rows * transformed_cols);
        }
    }

/*
    fprintf(stderr, "MULTIPLIED\n");
    cuComplex mul[num_images][num_kernels][padded_rows][transformed_cols];
    cudaMemcpy(mul, multiplied, sizeof(cufftComplex) * num_images * num_kernels * padded_rows * transformed_cols, cudaMemcpyDeviceToHost);
    for(uint32 image_index = 0; image_index < num_images; image_index++) {
        for(uint32 kernel_index = 0; kernel_index < num_kernels; kernel_index++) {
            for(uint32 r = 0; r < padded_rows; r++) {
                for(uint32 c = 0; c < transformed_cols; c++) {
                    fprintf(stderr,
                            "%.0f,%.0f ",
                            cuCrealf(mul[image_index][kernel_index][r][c]),
                            cuCimagf(mul[image_index][kernel_index][r][c]));
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n\n--------------------\n\n");
    }
*/

    // do inverse fft
    cufftReal *inverse_transformed;
    cudaMalloc((void**)&inverse_transformed, sizeof(cufftReal) * num_images * num_kernels * padded_rows * padded_cols);

    cufftExecC2R(inv_plan, multiplied, inverse_transformed);

    // scale the results appropriately (cufft does non-normalized transforms)
    blocks_per_grid = (num_images * num_kernels * padded_rows * padded_cols + threads_per_block - 1) / threads_per_block;
    elementwise_vector_scalar_multiply_inplace<<<blocks_per_grid, threads_per_block>>>(inverse_transformed,
                                                                                       1.0f / (padded_rows * padded_cols),
                                                                                       num_images * num_kernels * padded_rows * padded_cols);


    cudaFree(fft_input);
    cudaFree(transformed);
    cudaFree(multiplied);
    cudaFree(inverse_transformed);
    ////////////////////////////////////////////////////////////////////
    } // end timing-iteration for loop

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


/*
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
*/

    // TODO: Other cleanup -- memory freeing, etc.
    cufftDestroy(fwd_plan); // TODO: reuse fft plans
    cufftDestroy(inv_plan);
}
