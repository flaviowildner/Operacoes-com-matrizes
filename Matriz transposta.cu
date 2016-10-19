#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


__global__ void Transposta(float *c, float *r, int n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; 

	if (x >= n)
		return;

	for (int y = 0; y < n; ++y)
	{
		int from = x + y * n;
		int to = y + x * n;

		r[to] = c[from];
	}
}

void imprimir(float *c, float *r, int n){
	int i = 0, j = 0;
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			printf("%f ", c[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			printf("%f ", r[i*n + j]);
		}
		printf("\n");
	}
}


int main()
{
	cudaDeviceReset();

	//Variáveis
	FILE *cFile;
	int i;
	int n = 0;
	int size = n*n;

	int larguraC, alturaC;


	cFile = fopen("MatrizC.dat", "rb");
	if (cFile == NULL) { fputs("File error", stderr); exit(1); }


	fread(&larguraC, sizeof(int), 1, cFile);
	fread(&alturaC, sizeof(int), 1, cFile);

	n = alturaC;
	size = n*n;


	//Matrizes
	float *c, *r;
	float *d_c, *d_r;


	//Declarando variáveis inciais
	c = (float *)malloc(size * sizeof(float));
	r = (float *)malloc(size * sizeof(float));


	//Declarando variáveis na memória da GPU
	cudaMalloc((void **)&d_c, size*sizeof(float));
	cudaMalloc((void **)&d_r, size*sizeof(float));


	//Gerar Matrizes
	for (i = 0; i < size; i++){
		fread(&c[i], sizeof(float), 1, cFile);
	}

	/*for (i = 0; i < size; i++){
		c[i] = i;
	}*/


	cudaMemcpy(d_c, c, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, r, size*sizeof(float), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(size, 1, 1);
	dim3 dimGrid(1, 1, 1);

	if (size > 512){
		threadsPerBlock.x = 256;
		threadsPerBlock.y = 1;
		threadsPerBlock.z = 1;
		dimGrid.x = size / threadsPerBlock.x;
		dimGrid.y = 1;
		dimGrid.z = 1;
	}

	//Chama função no KERNEL
	Transposta<<<dimGrid, threadsPerBlock>>>(d_c, d_r, n);

	cudaDeviceSynchronize();


	cudaMemcpy(r, d_r, size*sizeof(float), cudaMemcpyDeviceToHost);

	//imprimir(c, r, n);




	//fclose(cFile);
	free(c);
	free(r);
	cudaFree(d_c);
	cudaFree(d_r);


	system("pause");


	return 0;
}