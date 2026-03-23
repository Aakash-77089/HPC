{
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!nvidia-smi"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "slDbOqYQGVaZ",
        "outputId": "5c843d9b-82b8-4814-955e-3c7301c3e9e3"
      },
      "id": "slDbOqYQGVaZ",
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mon Mar 23 05:10:06 2026       \n",
            "+-----------------------------------------------------------------------------------------+\n",
            "| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |\n",
            "+-----------------------------------------+------------------------+----------------------+\n",
            "| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n",
            "| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n",
            "|                                         |                        |               MIG M. |\n",
            "|=========================================+========================+======================|\n",
            "|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |\n",
            "| N/A   39C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |\n",
            "|                                         |                        |                  N/A |\n",
            "+-----------------------------------------+------------------------+----------------------+\n",
            "\n",
            "+-----------------------------------------------------------------------------------------+\n",
            "| Processes:                                                                              |\n",
            "|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n",
            "|        ID   ID                                                               Usage      |\n",
            "|=========================================================================================|\n",
            "|  No running processes found                                                             |\n",
            "+-----------------------------------------------------------------------------------------+\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install nvcc4jupyter"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tYsF-h_vGeMo",
        "outputId": "9ed65298-bd7a-4816-e419-41aec53f565e"
      },
      "id": "tYsF-h_vGeMo",
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting nvcc4jupyter\n",
            "  Downloading nvcc4jupyter-1.2.1-py3-none-any.whl.metadata (5.1 kB)\n",
            "Downloading nvcc4jupyter-1.2.1-py3-none-any.whl (10 kB)\n",
            "Installing collected packages: nvcc4jupyter\n",
            "Successfully installed nvcc4jupyter-1.2.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile scalar_mul.cu\n",
        "#include <stdio.h>\n",
        "\n",
        "#define N 512\n",
        "\n",
        "__global__ void\n",
        "scalarMultiply(float *arr, float k) {\n",
        "    int idx = threadIdx.x + blockIdx.x * blockDim.x;\n",
        "    arr[idx] *= k;\n",
        "}\n",
        "\n",
        "int main() {\n",
        "    float h_arr[N];\n",
        "    float k = 2.5;\n",
        "\n",
        "    for (int i = 0; i < N; i++) {\n",
        "        h_arr[i] = i + 1;\n",
        "    }\n",
        "\n",
        "    float *d_arr;\n",
        "\n",
        "    // Allocate Memory on GPU\n",
        "    cudaMalloc((void**)&d_arr, N * sizeof(float));\n",
        "\n",
        "    // Copy Memory from CPU to GPU\n",
        "    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);\n",
        "\n",
        "    // Defining Execution Configuration\n",
        "    int threadsperBlock = 256;\n",
        "    int blockperGrid = (N + threadsperBlock - 1) / threadsperBlock;\n",
        "\n",
        "    // Launch Kernel\n",
        "\n",
        "    scalarMultiply<<<blockperGrid, threadsperBlock>>>(d_arr, k);\n",
        "\n",
        "    //wait for GPU to finish\n",
        "    cudaDeviceSynchronize();\n",
        "\n",
        "    // copy result back to cpu\n",
        "    cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);\n",
        "\n",
        "    printf(\"Result (first 10 elements) : \\n\");\n",
        "    for ( int i = 0; i < 10; i++) {\n",
        "        printf(\"%.2f\", h_arr[i]);\n",
        "    }\n",
        "    printf(\"\\n\");\n",
        "\n",
        "    // free GPU memory\n",
        "    cudaFree(d_arr);\n",
        "\n",
        "    return 0;\n",
        " }"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a2cvL2pNEfrV",
        "outputId": "c9d0d635-3764-4b3f-d2ad-bca93fa82e82"
      },
      "id": "a2cvL2pNEfrV",
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting scalar_mul.cu\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!nvcc scalar_mul.cu -o scalar_mul"
      ],
      "metadata": {
        "id": "cNytAI3NKilI",
        "outputId": "976f7ea8-df57-472f-eb19-03f69cf1482f",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "id": "cNytAI3NKilI",
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!./scalar_mul"
      ],
      "metadata": {
        "id": "w652TCGFKqWR",
        "outputId": "e7564817-f0a5-4d97-fb76-3d8b88543520",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "id": "w652TCGFKqWR",
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Result (first 10 elements) : \n",
            "2.505.007.5010.0012.5015.0017.5020.0022.5025.00\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    },
    "required_libs": [],
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 5
}