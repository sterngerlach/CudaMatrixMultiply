
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
    /* 各ブロックでは, 行列Cの各部分行列の計算を実行 */
    /* 各スレッドでは, 行列Cの部分行列内の, 各要素の計算を実行 */

    /* 行列Cの部分行列のインデックス */
    int bx = blockIdx.x;
    int by = blockIdx.y;

    /* 行列Cの部分行列内の, 各要素のインデックス */
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    /* 行列Aの最初の部分行列へのインデックス */
    /* beginAは行列Aの(BlockSize * by, 0)要素 */
    int beginA = colA * (BlockSize * by);

    /* 行列Aの部分行列へのインデックスの最大値 */
    /* endAは行列Aの(Blocksize * by, colA - 1)要素 */
    int endA = beginA + colA - 1;
    
    /* 行列Aの部分行列単位のステップ */
    /* stepAは部分行列の大きさ分だけ横に移動させる */
    int stepA = BlockSize;

    /* 行列Bの最初の部分行列へのインデックス */
    /* beginBは行列Bの(0, BlockSize * bx)要素 */
    int beginB = (BlockSize * bx);

    /* 行列Bの部分行列へのインデックスの最大値 */
    /* endBは行列Bの(rowB - 1, BlockSize * bx)要素 */
    /* int endB = colB * (rowB - 1) + (BlockSize * bx); */

    /* 行列Bの部分行列単位のステップ */
    /* stepBは部分行列の大きさ分だけ縦に移動させる */
    int stepB = colB * BlockSize;

    /* 行列Cの部分行列の各要素の計算結果 */
    float subC = 0.0f;

    /* 行列Cの部分行列の計算を実行 */
    for (int a = beginA, b = beginB; a <= endA; a += stepA, b += stepB) {
        /* 行列AとBの部分行列同士の乗算を繰り返して, 行列Cの部分行列を計算
         * aとbは行列AとBの部分行列の先頭へのインデックス
         * stepAとstepBずつインデックスを加算していくことで,
         * 行列AとBの次の部分行列の先頭を参照できるようにする */

        /* ブロック内の各スレッドが共有する部分行列 */
        __shared__ float subMatA[BlockSize][BlockSize];
        __shared__ float subMatB[BlockSize][BlockSize];

        /* ブロック内の各スレッドが, デバイスメモリから共有メモリに,
         * 部分行列の各要素を転送 */
        subMatA[ty][tx] = matA[a + colA * ty + tx];
        subMatB[ty][tx] = matB[b + colB * ty + tx];

        /* 部分行列が共有メモリに転送されるように, スレッド間での同期をとる */
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BlockSize; ++k)
            subC += subMatA[ty][k] * subMatB[k][tx];

        /* 行列AとBの次の部分行列を読み込む前に, スレッド間での同期をとる */
        __syncthreads();
    }

    /* 各スレッドが行列Cの各要素に書き込み */
    int c = colC * (BlockSize * by) + (BlockSize * bx);
    matC[c + colC * ty + tx] = subC;
}

int main(int argc, char** argv)
{
    const unsigned int BlockSize = 32;
    
    /* 行列の行数と列数の設定 */
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

    /* 行列用のメモリ領域を確保 */
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

    /* 行列AとBを初期化 */
    std::fill(hostMatA, hostMatA + rowA * colA, 1.0f);
    std::fill(hostMatB, hostMatB + rowB * colB, valueB);

    cudaError_t cudaErr = cudaError::cudaSuccess;

    /* デバイスの行列用のメモリ領域を確保 */
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

    /* 行列AとBをホストからデバイスに転送 */
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

    /* デバイス上でベクトルの乗算を実行 */
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

    /* 計算が終わるまで待機 */
    cudaErr = ::cudaDeviceSynchronize();

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize() failed: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    /* 計算結果をデバイスからホストに転送 */
    cudaErr = ::cudaMemcpy(hostMatC, deviceMatC, memSizeC, cudaMemcpyDeviceToHost);

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to copy matrix C from device to host: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    /* 計算結果の検証 */
    for (unsigned int i = 0; i < rowC * colC; ++i) {
        /* 相対誤差の計算 */
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
    /* デバイスの行列用のメモリ領域を解放 */
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

    /* 行列用のメモリ領域を解放 */
    if (hostMatA != nullptr)
        delete[] hostMatA;

    if (hostMatB != nullptr)
        delete[] hostMatB;

    if (hostMatC != nullptr)
        delete[] hostMatC;

    return EXIT_SUCCESS;
}
