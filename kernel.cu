
/* kernel.cu */

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <int BlockSize>
__global__ void MultiplyMatrix(
    float* matC, float* matA, float* matB,
    int colA, int colB, int colC)
{
    /* �e�u���b�N�ł�, �s��C�̊e�����s��̌v�Z�����s */
    /* �e�X���b�h�ł�, �s��C�̕����s�����, �e�v�f�̌v�Z�����s */

    /* �s��C�̕����s��̃C���f�b�N�X */
    int bx = blockIdx.x;
    int by = blockIdx.y;

    /* �s��C�̕����s�����, �e�v�f�̃C���f�b�N�X */
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    /* �s��A�̍ŏ��̕����s��ւ̃C���f�b�N�X */
    /* beginA�͍s��A��(BlockSize * by, 0)�v�f */
    int beginA = colA * (BlockSize * by);

    /* �s��A�̕����s��ւ̃C���f�b�N�X�̍ő�l */
    /* endA�͍s��A��(Blocksize * by, colA - 1)�v�f */
    int endA = beginA + colA - 1;
    
    /* �s��A�̕����s��P�ʂ̃X�e�b�v */
    /* stepA�͕����s��̑傫�����������Ɉړ������� */
    int stepA = BlockSize;

    /* �s��B�̍ŏ��̕����s��ւ̃C���f�b�N�X */
    /* beginB�͍s��B��(0, BlockSize * bx)�v�f */
    int beginB = (BlockSize * bx);

    /* �s��B�̕����s��ւ̃C���f�b�N�X�̍ő�l */
    /* endB�͍s��B��(rowB - 1, BlockSize * bx)�v�f */
    /* int endB = colB * (rowB - 1) + (BlockSize * bx); */

    /* �s��B�̕����s��P�ʂ̃X�e�b�v */
    /* stepB�͕����s��̑傫���������c�Ɉړ������� */
    int stepB = colB * BlockSize;

    /* �s��C�̕����s��̊e�v�f�̌v�Z���� */
    float subC = 0.0f;

    /* �s��C�̕����s��̌v�Z�����s */
    for (int a = beginA, b = beginB; a <= endA; a += stepA, b += stepB) {
        /* �s��A��B�̕����s�񓯎m�̏�Z���J��Ԃ���, �s��C�̕����s����v�Z
         * a��b�͍s��A��B�̕����s��̐擪�ւ̃C���f�b�N�X
         * stepA��stepB���C���f�b�N�X�����Z���Ă������Ƃ�,
         * �s��A��B�̎��̕����s��̐擪���Q�Ƃł���悤�ɂ��� */

        /* �u���b�N���̊e�X���b�h�����L���镔���s�� */
        __shared__ float subMatA[BlockSize][BlockSize];
        __shared__ float subMatB[BlockSize][BlockSize];

        /* �u���b�N���̊e�X���b�h��, �f�o�C�X���������狤�L��������,
         * �����s��̊e�v�f��]�� */
        subMatA[ty][tx] = matA[a + colA * ty + tx];
        subMatB[ty][tx] = matB[b + colB * ty + tx];

        /* �����s�񂪋��L�������ɓ]�������悤��, �X���b�h�Ԃł̓������Ƃ� */
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BlockSize; ++k)
            subC += subMatA[ty][k] * subMatB[k][tx];

        /* �s��A��B�̎��̕����s���ǂݍ��ޑO��, �X���b�h�Ԃł̓������Ƃ� */
        __syncthreads();
    }

    /* �e�X���b�h���s��C�̊e�v�f�ɏ������� */
    int c = colC * (BlockSize * by) + (BlockSize * bx);
    matC[c + colC * ty + tx] = subC;
}

int main(int argc, char** argv)
{
    const unsigned int BlockSize = 32;
    
    /* �s��̍s���Ɨ񐔂̐ݒ� */
    const float valueB = 0.01f;
    const unsigned int rowA = 960;
    const unsigned int colA = 480;
    const unsigned int rowB = 480;
    const unsigned int colB = 960;

    assert(colA == rowB);
    assert(rowA % BlockSize == 0);
    assert(colA % BlockSize == 0);
    assert(rowB % BlockSize == 0);
    assert(colB % BlockSize == 0);

    unsigned int rowC = rowA;
    unsigned int colC = colB;

    unsigned int memSizeA = sizeof(float) * rowA * colA;
    unsigned int memSizeB = sizeof(float) * rowB * colB;
    unsigned int memSizeC = sizeof(float) * rowC * colC;

    std::cout << "matrix multiplication: "
              << "A(" << rowA << ", " << colA << ") * "
              << "B(" << rowB << ", " << colB << ")\n";

    /* �s��p�̃������̈���m�� */
    float* hostMatA = new (std::nothrow) float[rowA * colA];

    if (hostMatA == nullptr) {
        std::cerr << "failed to allocate sufficient memory for matrix A\n";
        goto Cleanup;
    }

    float* hostMatB = new (std::nothrow) float[rowB * colB];

    if (hostMatB == nullptr) {
        std::cerr << "failed to allocate sufficient memory for matrix B\n";
        goto Cleanup;
    }

    float* hostMatC = new (std::nothrow) float[rowC * colC];

    if (hostMatC == nullptr) {
        std::cerr << "failed to allocate sufficient memory for matrix C\n";
        goto Cleanup;
    }

    /* �s��A��B�������� */
    std::fill(hostMatA, hostMatA + rowA * colA, 1.0f);
    std::fill(hostMatB, hostMatB + rowB * colB, valueB);

    cudaError_t cudaErr = cudaError::cudaSuccess;

    /* �f�o�C�X�̍s��p�̃������̈���m�� */
    float* deviceMatA = nullptr;
    cudaErr = ::cudaMalloc(&deviceMatA, memSizeA);

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to allocate device matrix A: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    float* deviceMatB = nullptr;
    cudaErr = ::cudaMalloc(&deviceMatB, memSizeB);

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to allocate device matrix B: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    float* deviceMatC = nullptr;
    cudaErr = ::cudaMalloc(&deviceMatC, memSizeC);

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to allocate device matrix C: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    /* �s��A��B���z�X�g����f�o�C�X�ɓ]�� */
    cudaErr = ::cudaMemcpy(deviceMatA, hostMatA, memSizeA, cudaMemcpyHostToDevice);

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to copy matrix A from host to device: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    cudaErr = ::cudaMemcpy(deviceMatB, hostMatB, memSizeB, cudaMemcpyHostToDevice);

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to copy matrix B from host to device: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    /* �f�o�C�X��Ńx�N�g���̏�Z�����s */
    dim3 dimBlock { BlockSize, BlockSize, 1 };
    dim3 dimGrid { colC / dimBlock.x, rowC / dimBlock.y, 1 };

    MultiplyMatrix<BlockSize><<<dimGrid, dimBlock>>>(
        deviceMatC, deviceMatA, deviceMatB, colA, colB, colC);
    cudaErr = ::cudaGetLastError();

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to launch MultiplyMatrix kernel: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    /* �v�Z���I���܂őҋ@ */
    cudaErr = ::cudaDeviceSynchronize();

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize() failed: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    /* �v�Z���ʂ��f�o�C�X����z�X�g�ɓ]�� */
    cudaErr = ::cudaMemcpy(hostMatC, deviceMatC, memSizeC, cudaMemcpyDeviceToHost);

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to copy matrix C from device to host: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    /* �v�Z���ʂ̌��� */
    for (unsigned int i = 0; i < rowC * colC; ++i) {
        /* ���Ό덷�̌v�Z */
        double absErr = std::fabs(hostMatC[i] - (colA * valueB));
        double absValue = std::fabs(hostMatC[i]);
        double dotLength = static_cast<double>(rowA);
        double relErr = absErr / absValue / dotLength;

        if (relErr > 1e-6) {
            std::cerr << "result verification failed at element ("
                      << (i / colC) << ", " << (i % colC) << ")\n";
            goto Cleanup;
        }
    }

    std::cout << "matrix multiplication succeeded\n";

Cleanup:
    /* �f�o�C�X�̍s��p�̃������̈����� */
    if (deviceMatA != nullptr) {
        cudaErr = ::cudaFree(deviceMatA);

        if (cudaErr != cudaError::cudaSuccess)
            std::cerr << "failed to free device matrix A: "
                      << ::cudaGetErrorString(cudaErr) << '\n';
    }

    if (deviceMatB != nullptr) {
        cudaErr = ::cudaFree(deviceMatB);

        if (cudaErr != cudaError::cudaSuccess)
            std::cerr << "failed to free device matrix A: "
            << ::cudaGetErrorString(cudaErr) << '\n';
    }

    if (deviceMatC != nullptr) {
        cudaErr = ::cudaFree(deviceMatC);

        if (cudaErr != cudaError::cudaSuccess)
            std::cerr << "failed to free device matrix A: "
            << ::cudaGetErrorString(cudaErr) << '\n';
    }

    /* �s��p�̃������̈����� */
    if (hostMatA != nullptr)
        delete[] hostMatA;

    if (hostMatB != nullptr)
        delete[] hostMatB;

    if (hostMatC != nullptr)
        delete[] hostMatC;

    return EXIT_SUCCESS;
}
