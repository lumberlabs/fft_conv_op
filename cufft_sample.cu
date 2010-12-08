// build with nvcc cufft_sample.cu -lcufft

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#define DEBUG 0

// TODO: Why are these necessary? What's wrong with stdint.h??
typedef unsigned int uint32;
typedef int int32;

// TODO: is it possible to optimize this?
__global__ void pad_images_and_kernels(cufftReal *images,
                                       cufftReal *kernels,
                                       cufftReal *dest,
                                       uint32 total_images,
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

    cufftReal out;
    if(row_index >= source_rows || col_index >= source_cols) {
        out = 0.0f;
    } else {
        out = src[src_offset * source_rows * source_cols + row_index * source_cols + col_index];
    }
    
    dest[ik_offset * padded_rows * padded_cols + row_index * padded_cols + col_index] = out;
}


// TODO: is it possible to optimize this? reduce access to global mem by loading into shared mem, etc.
// TODO: what happens when element_length is bigger than the allowed num threads?
__global__ void elementwise_image_kernel_multiply(cufftComplex *transformed,
                                                  cufftComplex *multiplied,
                                                  uint32 batch_size,
                                                  uint32 num_kernels,
                                                  uint32 num_images,
                                                  uint32 element_length) {
    uint32 batch_kernel_index = blockIdx.x;
    uint32 batch_index = batch_kernel_index / num_kernels;
    uint32 kernel_index = batch_kernel_index % num_kernels;
    uint32 image_index = blockIdx.y;
    uint32 element_index = threadIdx.x;
    
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

// TODO: is it possible to optimize this?
__global__ void add_across_images_and_normalize(cufftReal *inverse_transformed,
                                                cufftReal *added,
                                                uint32 num_images,
                                                uint32 batch_size,
                                                uint32 num_kernels,
                                                uint32 padded_rows,
                                                uint32 padded_cols,
                                                float normalization_factor) {
    uint32 row = threadIdx.x;
    uint32 col = threadIdx.y;
    uint32 batch_index = blockIdx.x;
    uint32 kernel_index = blockIdx.y;
    float sum = 0.0f;
    for(uint32 image_index = 0; image_index < num_images; image_index++) {
        cufftReal *image = inverse_transformed + (batch_index * num_kernels * num_images + kernel_index * num_images + image_index) * padded_rows * padded_cols;
        cufftReal *image_element = image + row * padded_cols + col;
        sum += *image_element;
    }
    cufftReal *added_destination = added + batch_index * num_kernels * padded_rows * padded_cols + kernel_index * padded_rows * padded_cols + row * padded_cols + col;
    *added_destination = sum / normalization_factor;
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
int main(int argc, char *argv[]) {
#if RUN_SPEED_TESTS
    uint32 batch_size = 500;
    uint32 num_images = 1;
    uint32 image_rows = 28;
    uint32 image_cols = 28;
    uint32 num_kernels = 50;
    uint32 kernel_rows = 5;
    uint32 kernel_cols = 5;
#else
    uint32 batch_size = 1;
    uint32 num_images = 4;
    uint32 image_rows = 4;
    uint32 image_cols = 4;
    uint32 num_kernels = 2;
    uint32 kernel_rows = 2;
    uint32 kernel_cols = 2;
#endif

    uint32 num_padded = batch_size * num_images + num_kernels * num_images; // total images + total kernels
    uint32 convolved_rows = image_rows + kernel_rows - 1;
    uint32 convolved_cols = image_cols + kernel_cols - 1;
    uint32 padded_rows = next_power_of_two(convolved_rows);
    uint32 padded_cols = next_power_of_two(convolved_cols);
    uint32 transformed_cols = padded_cols / 2 + 1; // only non-redundant complex coefficients are calculated

    // set up images and kernels
    cufftReal *images = (cufftReal *)malloc(sizeof(cufftReal) * batch_size * num_images * image_rows * image_cols);
    //cufftReal images[batch_size][num_images][image_rows][image_cols];
    for(uint32 b = 0; b < batch_size; b++) {
        for(uint32 i = 0; i < num_images; i++) {
            for(uint32 r = 0; r < image_rows; r++) {
                for(uint32 c = 0; c < image_cols; c++) {
                    // images[b][i][r][c] = i + 1;
                    cufftReal *image_ptr = images
                                         + b * num_images * image_rows * image_cols
                                         + i * image_rows * image_cols
                                         + r * image_cols
                                         + c;
                    *image_ptr = i + 1;
                }
            }
        }
    }
    
#if RUN_SPEED_TESTS
#else
    fprintf(stderr, "INBOUND IMAGES\n-----------------\n");
    for(uint32 b = 0; b < batch_size; b++) {
        for(uint32 i = 0; i < num_images; i++) {
            fprintf(stderr, "<image b %i, i %i>\n", b, i);
            for(uint32 r = 0; r < image_rows; r++) {
                for(uint32 c = 0; c < image_cols; c++) {
                    fprintf(stderr, "%.0f ", images[b][i][r][c]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
    }
#endif

    cufftReal *kernels = (cufftReal *)malloc(sizeof(cufftReal) * num_kernels * num_images * kernel_rows * kernel_cols);
    //cufftReal kernels[num_kernels][num_images][kernel_rows][kernel_cols];
    for(uint32 k = 0; k < num_kernels; k++) {
        for(uint32 i = 0; i < num_images; i++) {
            for(uint32 r = 0; r < kernel_rows; r++) {
                for(uint32 c = 0; c < kernel_cols; c++) {
                    // kernels[k][i][r][c] = (k + 1) * (i + 1);
                    cufftReal *kernel_ptr = kernels
                                          + k * num_images * kernel_rows * kernel_cols
                                          + i * kernel_rows * kernel_cols
                                          + r * kernel_cols
                                          + c;
                    *kernel_ptr = (k + 1) * (i + 1);
                }
            }
        }
    }

#if RUN_SPEED_TESTS
#else
    fprintf(stderr, "INBOUND KERNELS\n");
    for(uint32 k = 0; k < num_kernels; k++) {
        for(uint32 i = 0; i < num_images; i++) {
            fprintf(stderr, "<kernel k %i, i %i>\n", k, i);
            for(uint32 r = 0; r < kernel_rows; r++) {
                for(uint32 c = 0; c < kernel_cols; c++) {
                    fprintf(stderr, "%.0f ", kernels[k][i][r][c]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
    }
#endif

    // copy images and kernels to device
    cufftReal *inbound_images;
    uint32 inbound_images_size = sizeof(cufftReal) * batch_size * num_images * image_rows * image_cols;
    cudaMalloc((void **)&inbound_images, inbound_images_size);
    cudaMemcpy(inbound_images, images, inbound_images_size, cudaMemcpyHostToDevice);

    cufftReal *inbound_kernels;
    uint32 inbound_kernels_size = sizeof(cufftReal) * num_kernels * num_images * kernel_rows * kernel_cols;
    cudaMalloc((void **)&inbound_kernels, inbound_kernels_size);
    cudaMemcpy(inbound_kernels, kernels, inbound_kernels_size, cudaMemcpyHostToDevice);

    // assume we can pay the planning price just once and amortize it away, so do the planning up front
    int32 padded_dimensions[2] = {padded_rows, padded_cols};

    cufftHandle fwd_plan;
    cufftResult plan_result = cufftPlanMany(&fwd_plan, // plan
                                            2, // rank
                                            padded_dimensions, // dimensions
                                            NULL, 1, 0, NULL, 1, 0, // boilerplate for contiguous access (non-contiguous access not supported now)
                                            CUFFT_R2C, // fwd transform, real to complex
                                            num_padded // fft batch size
                                           );
    cufftSetCompatibilityMode(fwd_plan, CUFFT_COMPATIBILITY_NATIVE); // performance only
    
    cufftHandle inv_plan;
    cufftPlanMany(&inv_plan, // plan
                  2, // rank
                  padded_dimensions, // dimensions
                  NULL, 1, 0, NULL, 1, 0, // boilerplate for contiguous access (non-contiguous access not supported now)
                  CUFFT_C2R, // inv transform, complex to real
                  batch_size * num_kernels * num_images // ifft batch size
                 );
     // CUFFT_COMPATIBILITY_NATIVE needed to prevent extra padding, so output is compact
     // and nicely accessible via c pointer arithmetic
     cufftSetCompatibilityMode(inv_plan, CUFFT_COMPATIBILITY_NATIVE);

#if RUN_SPEED_TESTS
    // done with setup. this is presumably where we'd be at the beginning of the conv op
    // in theano, so we can start timing here if we want
    uint32 num_iterations = 1000;
    struct timeval start;
    gettimeofday(&start, NULL);
    for(uint32 iteration = 0; iteration < num_iterations; iteration++) {
#endif

    // rearrange images and kernels to their new padded size, all contiguous
    // to each other, since that is what the batched fft requires right now
    
    cufftReal *fft_input;
    cudaMalloc((void **)&fft_input, sizeof(cufftReal) * num_padded * padded_rows * padded_cols);

    dim3 padding_threads(padded_rows, padded_cols); // TODO: how to handle padded_rows * padded_cols > 1024?
    pad_images_and_kernels<<<num_padded, padding_threads>>>(inbound_images,
                                                            inbound_kernels,
                                                            fft_input,
                                                            batch_size * num_images, // number of images in the fft
                                                            image_rows,
                                                            image_cols,
                                                            kernel_rows,
                                                            kernel_cols,
                                                            padded_rows,
                                                            padded_cols);

#if DEBUG
    fprintf(stderr, "PADDED\n");
    cufftReal pad[num_padded][padded_rows][padded_cols];
    cudaMemcpy(pad, fft_input, sizeof(cufftReal) * num_padded * padded_rows * padded_cols, cudaMemcpyDeviceToHost);
    for(uint32 n = 0; n < num_padded; n++) {
        for(uint32 r = 0; r < padded_rows; r++) {
            for(uint32 c = 0; c < padded_cols; c++) {
                fprintf(stderr,
                        "%.0f ", pad[n][r][c]);
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }
#endif
    
    // perform forward fft

    cufftComplex *transformed;
    cudaMalloc((void **)&transformed, sizeof(cufftComplex) * num_padded * padded_rows * transformed_cols);

    cufftExecR2C(fwd_plan, fft_input, transformed);

    // do elemwise multiplication
    cufftComplex *multiplied;
    // this memory will be re-used when the C2R transform is done in place, so make sure there's enough space
    uint32 multiplied_size = sizeof(cufftComplex) * batch_size * num_kernels * num_images * padded_rows * transformed_cols;
    uint32 inverse_transformed_size = sizeof(cufftReal) * batch_size * num_kernels * num_images * padded_rows * padded_cols;
    cudaMalloc((void **)&multiplied, max(multiplied_size, inverse_transformed_size));


    dim3 dim_grid(batch_size * num_kernels, num_images);
    elementwise_image_kernel_multiply<<<dim_grid, padded_rows * transformed_cols>>>(transformed,
                                                                                    multiplied,
                                                                                    batch_size,
                                                                                    num_kernels,
                                                                                    num_images,
                                                                                    padded_rows * transformed_cols);

    // do inverse fft
    cufftReal *inverse_transformed;
    inverse_transformed = (cufftReal *)multiplied;

    cufftExecC2R(inv_plan, multiplied, inverse_transformed);

#if DEBUG
    fprintf(stderr, "INVERSE_TRANSFORMED\n");
    cufftReal inv[batch_size][num_kernels][num_images][padded_rows][padded_cols];
    cudaMemcpy(inv, inverse_transformed, sizeof(cufftReal) * batch_size * num_kernels * num_images * padded_rows * padded_cols, cudaMemcpyDeviceToHost);
    for(uint32 b = 0; b < batch_size; b++) {
        for(uint32 k = 0; k < num_kernels; k++) {
            for(uint32 i = 0; i < num_images; i++) {
                fprintf(stderr, "<trans b %i, k %i, i %i>\n", b, k, i);
                for(uint32 r = 0; r < padded_rows; r++) {
                    for(uint32 c = 0; c < padded_cols; c++) {
                        fprintf(stderr,
                                "%.0f ", inv[b][k][i][r][c]);
                    }
                    fprintf(stderr, "\n");
                }
                fprintf(stderr, "\n");
            }
        }
    }
#endif


    cufftReal *added;
    cudaMalloc((void **)&added, sizeof(cufftReal) * batch_size * num_kernels * padded_rows * padded_cols);
    // sum across images and scale the results appropriately (cufft does non-normalized transforms)
    dim3 adding_grid(batch_size, num_kernels);
    dim3 adding_threads(padded_rows, padded_cols);
    add_across_images_and_normalize<<<adding_grid, adding_threads>>>(inverse_transformed,
                                                                     added,
                                                                     num_images,
                                                                     batch_size,
                                                                     num_kernels,
                                                                     padded_rows,
                                                                     padded_cols,
                                                                     padded_rows * padded_cols); // normalization factor

    cudaFree(fft_input);
    cudaFree(transformed);
    cudaFree(multiplied);
#if RUN_SPEED_TESTS
    cudaFree(added);
#endif

#if RUN_SPEED_TESTS
    } // end timing-iteration for loop
    struct timeval end;
    gettimeofday(&end, NULL);
#endif

#if RUN_SPEED_TESTS
    unsigned long elapsed_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    float elapsed_s = elapsed_us / 1000000.0f;
    fprintf(stderr,
            "batch size: %i, num_images: %i, image size: %ix%i, num_kernels: %i, kernel size: %ix%i\n%.3fs elapsed for %i iterations (%f s/iter)\n",
            batch_size,
            num_images,
            image_rows,
            image_cols,
            num_kernels,
            kernel_rows,
            kernel_cols,
            elapsed_s,
            num_iterations,
            elapsed_s / (float)num_iterations);
#else
    // TODO: Set strides appropriately (or do memcpys to get rid of unneeded padding)
    float results[batch_size][num_kernels][padded_rows][padded_cols];
    // copy results back to host
    cudaMemcpy(results, added, sizeof(cufftReal) * batch_size * num_kernels * padded_rows * padded_cols, cudaMemcpyDeviceToHost);

    fprintf(stderr, "OUTBOUND\n");
    for(uint32 b = 0; b < batch_size; b++) {
        for(uint32 k = 0; k < num_kernels; k++) {
            fprintf(stderr, "<out b %i, k %i>\n", b, k);
            for(uint32 r = 0; r < convolved_rows; r++) {
                for(uint32 c = 0; c < convolved_cols; c++) {
                    fprintf(stderr, "%.0f ", results[b][k][r][c]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
    }
#endif

    cufftDestroy(fwd_plan);
    cufftDestroy(inv_plan);
    cudaFree(inbound_images);
    cudaFree(inbound_kernels);
    free(images);
    free(kernels);
}
