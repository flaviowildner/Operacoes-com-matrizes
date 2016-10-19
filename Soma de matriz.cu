#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>




__global__ void AddVet(float *a, float *b, float *c)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	c[index] = a[index] + b[index];
}

void imprimir(float *a, float *b, float *c, int n){
	int i = 0, j = 0;
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			printf("%f ", a[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			printf("%f ", b[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			printf("%f ", c[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");
}


int main()
{
	cudaDeviceReset();

	//Variáveis
	FILE *aFile;
	FILE *bFile;
	int i;
	int n;
	int size;

	int larguraA, alturaA, larguraB, alturaB;


	aFile = fopen("MatrizA.dat", "rb");
	if (aFile == NULL) { fputs("File error", stderr); exit(1); }

	bFile = fopen("MatrizB.dat", "rb");
	if (bFile == NULL) { fputs("File error", stderr); exit(1); }

	fread(&larguraA, sizeof(int), 1, aFile);
	fread(&alturaA, sizeof(int), 1, aFile);
	fread(&larguraB, sizeof(int), 1, bFile);
	fread(&alturaB, sizeof(int), 1, bFile);

	n = alturaA;
	size = n*n;
	
	

	//Matrizes
	float *a, *b, *c;
	float *d_a, *d_b, *d_c;


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





	cudaMemcpy(d_a, a, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, size*sizeof(float), cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(size, 1, 1);
	dim3 dimGrid(1, 1, 1);

	if (size > 512){
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 1;
		threadsPerBlock.z = 1;
		dimGrid.x = size / threadsPerBlock.x;
		dimGrid.y = 1;
		dimGrid.z = 1;
	}

	//Chama função no KERNEL
	AddVet<<<dimGrid, threadsPerBlock>>>(d_a, d_b, d_c);

	cudaDeviceSynchronize();


	cudaMemcpy(c, d_c, size*sizeof(float), cudaMemcpyDeviceToHost);

	//imprimir(a, b, c, n);

	printf("%f %f %f", a[1048575], b[1048575], c[1048575]);


	fclose(aFile);
	fclose(bFile);
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	system("pause");

	return 0;
}