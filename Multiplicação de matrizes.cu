#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>




__global__ void MultMatrix(float *a, float *b, float *c, int n)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int sum = 0;
	for (int k = 0; k < n; k++) {
		sum += a[row*n + k] * b[k * n + col];
	}
	c[row*n + col] = sum;
}



void imprimir(int *a, int *b, int *c, int n){

	int i = 0, j = 0;
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			printf("%d ", a[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			printf("%d ", b[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			printf("%d ", c[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");

}


int main()
{
	//Declarar ponteiros
	FILE *aFile;
	FILE *bFile;
	float *a, *b, *c;
	float *d_a, *d_b, *d_c;
	int i, j;
	int larguraA, alturaA, larguraB, alturaB;


	aFile = fopen("MatrizA.dat", "rb");
	if (aFile == NULL) { fputs("File error", stderr); exit(1); }

	bFile = fopen("MatrizB.dat", "rb");
	if (bFile == NULL) { fputs("File error", stderr); exit(1); }

	fread(&larguraA, sizeof(int), 1, aFile);
	fread(&alturaA, sizeof(int), 1, aFile);
	fread(&larguraB, sizeof(int), 1, bFile);
	fread(&alturaB, sizeof(int), 1, bFile);



	//Tamanhos
	const int n = alturaA;
	int size = n*n;
	int size_of = size*sizeof(float);

	//Reset GPU
	cudaDeviceReset();

	//Declarando variáveis inciais
	a = (float *)malloc(size * sizeof(float));
	b = (float *)malloc(size * sizeof(float));
	c = (float *)malloc(size * sizeof(float));


	//Declarando variáveis na memória da GPU
	cudaMalloc((void **)&d_a, size*sizeof(float));
	cudaMalloc((void **)&d_b, size*sizeof(float));
	cudaMalloc((void **)&d_c, size*sizeof(float));


	//Gerar Matrizes
	for (i = 0; i < size; i++){
		fread(&a[i], sizeof(float), 1, aFile);
	}
	for (i = 0; i < size; i++){
		fread(&b[i], sizeof(float), 1, bFile);
	}


	//Copiando dados do Host para o Device
	cudaMemcpy(d_a, a, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, size*sizeof(float), cudaMemcpyHostToDevice);



	dim3 threadsPerBlock(n, n, 1);
	dim3 dimGrid(1, 1, 1);

	if (n > 16){
		threadsPerBlock.x = 16;
		threadsPerBlock.y = 16;
		threadsPerBlock.z = 1;
		dimGrid.x = n / threadsPerBlock.x;
		dimGrid.y = n / threadsPerBlock.y;
		dimGrid.z = 1;
	}

	//Chama função no KERNEL
	MultMatrix<<<dimGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);


	//Sincronizando Threads
	cudaDeviceSynchronize();


	//Copiando o resultado(c) para o Host
	cudaMemcpy(c, d_c, size_of, cudaMemcpyDeviceToHost);


	//Testes
	//imprimir(a, b, c, n);


	//Free
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	system("pause");

	return 0;
}