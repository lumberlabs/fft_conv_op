#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <stdint.h>

// TODO: Why are these necessary? What's wrong with stdint.h?
typedef unsigned int uint32;

// TODO: optimize me
// optimizations include preloading (say) kernels and iterating over images
// etc. etc.
__global__ void elementwise_multiply(cufftComplex *src1, cufftComplex *src2, cufftComplex *dest, uint32 len) {
    uint32 index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < len) {
        dest[index] = cuCmulf(src1[index], src2[index]);
    }
}

uint32 next_power_of_two(uint32 i) {
    // from Sean Anderson's bit twiddling hacks (TODO: insert link)
    i--;
    i |= i >> 1;
    i |= i >> 2;
    i |= i >> 4;
    i |= i >> 8;
    i |= i >> 16;
    i++;
    return i;
}

int main(int argc, char *argv[])
{
    uint32 num_images = 2;
    uint32 image_rows = 28;
    uint32 image_cols = 28;

    uint32 num_kernels = 3;
    uint32 kernel_rows = 5;
    uint32 kernel_cols = 5;

    uint32 num_padded = num_images + num_kernels;
    uint32 padded_rows = next_power_of_two(image_rows + kernel_rows - 1);
    uint32 padded_cols = next_power_of_two(image_cols + kernel_cols - 1);

    fprintf(stderr, "padded cols: %u\n", padded_cols);

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

    // copy images and kernels to device
    // do this first, and simply (before padding etc., to mimic what theano
    // would have to do -- presumably, the data is already on the gpu)
    cufftReal *inbound_images;
    cudaMalloc((void**)&inbound_images, sizeof(cufftReal) * num_images * image_rows * image_cols);
    cudaMemcpy(images, inbound_images, sizeof(cufftReal) * num_images * image_rows * image_cols, cudaMemcpyHostToDevice);

    cufftReal *inbound_kernels;
    cudaMalloc((void**)&inbound_kernels, sizeof(cufftReal) * num_kernels * kernel_rows * kernel_cols);
    cudaMemcpy(kernels, inbound_kernels, sizeof(cufftReal) * num_kernels * kernel_rows * kernel_cols, cudaMemcpyHostToDevice);


    // rearrange images and kernels to their new padded size, all contiguous
    // to each other, since that is what the batched fft requires right now
    
    // TODO: Would writing a custom kernel to do this memory shuffling be faster?
    cufftReal *fft_input;
    cudaMalloc((void**)&fft_input, sizeof(cufftReal) * num_padded * padded_rows * padded_cols);
    cudaMemset(fft_input, sizeof(cufftReal) * num_padded * padded_rows * padded_cols, 0);
    for(int b = 0; b < num_images; b++) {
        for(int r = 0; r < image_rows; r++) {
            cudaMemcpy(fft_input + b * padded_rows * padded_cols + r * padded_rows,
                       inbound_images + b * image_rows * image_cols + r * image_cols,
                       sizeof(cufftReal) * image_cols,
                       cudaMemcpyDeviceToDevice);
        }
    }

    for(int b = 0; b < num_kernels; b++) {
        for(int r = 0; r < kernel_rows; r++) {
            cudaMemcpy(fft_input + (b + num_images) * padded_rows * padded_cols + r * padded_rows,
                       inbound_kernels + b * kernel_rows * kernel_cols + r * kernel_cols,
                       sizeof(cufftReal) * kernel_cols,
                       cudaMemcpyDeviceToDevice);
        }
    }
    
    
    // perform forward fft
    int padded_dimensions[2] = {padded_rows, padded_cols};

    cufftHandle fwd_plan;
    cufftPlanMany(&fwd_plan, // plan
                  2, // rank
                  padded_dimensions, // dimensions
                  NULL, 1, 0, NULL, 1, 0, // boilerplate for contiguous access (non-contiguous access not supported now)
                  CUFFT_R2C, // type
                  num_padded // batch size
                  );

    cufftComplex *transformed;
    cudaMalloc((void**)&transformed, sizeof(cufftComplex) * num_padded * padded_rows * padded_cols);

    cufftExecR2C(fwd_plan, fft_input, transformed);

    cufftDestroy(fwd_plan); // TODO: reuse fft plans

    // cudaFree(images); // TODO: memory cleanup as appropriate


    // do elemwise multiplication
    
    cufftComplex *multiplied;
    cudaMalloc((void**)&multiplied, sizeof(cufftComplex) * (num_images * num_kernels) * padded_rows * padded_cols);

    

/*
    // do inverse fft
    cufftHandle inv_plan;
    cufftPlanMany(&inv_plan, // plan
                  2, // rank
                  dimensions, // dimensions
                  NULL, 1, 0, NULL, 1, 0, // boilerplate for contiguous access (non-contiguous access not supported now)
                  CUFFT_C2R, // type
                  BATCHSIZE // batch size
                  );

    cufftReal *out_data;
    cudaMalloc((void**)&out_data, sizeof(cufftReal) * NX * NY * BATCHSIZE);

    cufftExecC2R(inv_plan, transformed_data, out_data);

    cufftDestroy(inv_plan);

    cudaFree(transformed_data);

    // copy results back to host
    cudaMemcpy(out_data, images, sizeof(cufftReal) * NX * NY, cudaMemcpyDeviceToHost);

    cudaFree(out_data);

    for(int x = 0; x < NX; x++) {
        for(int y = 0; y < NY; y++) {
            for(int b = 0; b < BATCHSIZE; b++) {
                if(images[x][y][b] != (float)b) {
                    fprintf(stderr, "Error at %d,%d,%d: %f\n", x, y, b, images[x][y][b]);
                }
            }
        }
    }
    */
}
