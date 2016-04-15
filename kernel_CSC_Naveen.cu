#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <time.h>

using namespace std;

cudaError_t spmv_csc_matrix_multi( int h_nOfCols, int *h_csc_row_a, int *h_csc_col_a, float *h_csc_val_a, 
								   float *h_csc_b, float *h_y, int nz, int nrows, int k);

__global__ void spmv_csc_matrix_kernel(int * dev_csc_row_a, float *dev_csc_val_a, int *dev_csc_col_a, 
								  float *dev_csc_b, float *dev_y,int nrows)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x ;
	
	if (i < nrows){
		dev_y[i] = 0.0;
				for (int j=dev_csc_col_a[i]; j<dev_csc_col_a[i+1]; ++j)
					dev_y[i] += dev_csc_val_a[j]*dev_csc_b[dev_csc_row_a[j]];	
	}
}
	

int main()
{
  	char name[100];	
    int nrows_a, ncols_a, nz_a,nrows_b, ncols_b, nz_b;
    int *R_A, *C_A,*R_B, *C_B;
	float *V_A, *V_B;
	clock_t t;
	int f;
	string filenameA[] = {"sM1000a.mtx","sM2000a.mtx","sM3000a.mtx" };
	string filenameB[] = {"sM1000b.mtx","sM2000b.mtx","sM3000b.mtx" };

	for( int f=0; f<3;f++)
	{
		ifstream infile;ifstream infile3;
		infile.open(filenameA[f]);
		//infile.open("sM1000a.mtx");	
		infile >> name; infile >> name; infile >> name ; infile >> name; infile >> name;
		infile >> nrows_a; infile >> ncols_a; infile >> nz_a;
		cout<<"Filename:"<<filenameA[f]<<"\tRows:"<<nrows_a<<"\tColumns:"<<ncols_a<<"\tNon Zeros:"<<nz_a<<endl;

		R_A = (int *) malloc(nz_a * sizeof(int));
		C_A = (int *) malloc(nz_a * sizeof(int));
		V_A = (float *) malloc(nz_a * sizeof(float));

		// 1 Dim Array
		int *csc_col_a = new int [nz_a];  // Extra Index for zero
		int *csc_row_a = new int [nz_a];  // Total number of NZ
		float *csc_val_a = new float [nz_a]; // Total number of NZ
	
		// Vector of Tuples 
		std::vector<std::tuple<int,int,float>> vectorA;

		for (int i=0; i<nz_a; i++){
			infile >> R_A[i]>>C_A[i]>>V_A[i];
			vectorA.push_back(std::make_tuple(C_A[i],R_A[i],V_A[i]));  // Sorted by Column first, then row, then value
		}
	
		infile.close();
		std::sort(begin(vectorA),end(vectorA));


		int col_check = 0;
		int nOfRows_a = 0; int nOfCols =0; 
		for (int l=0; l<nz_a; l++)
		{       
			if ( l == 0 ) {
				csc_col_a[l] = 0; // first index of the column array is always Zero
				col_check = get<0>(vectorA[l]);  // Take the new column for comparison 
			}
		
			if ( col_check!= get<0>(vectorA[l]))  // Take the new column for comparison 
			{
				col_check = get<0>(vectorA[l]); // change of column
				nOfCols++;
				csc_col_a[nOfCols] = l; 	// No. of rows in the coloumn
			}
			if((get<1>(vectorA[1]))>0)
				csc_row_a[l] = get<1>(vectorA[l])-1; // Get the row value. Reduce by 1
			else 
				csc_row_a[l] = get<1>(vectorA[l]); // Get the row value. Reduce by 1

			csc_val_a[l] = get<2>(vectorA[l]); // Get the value info
			if (l == nz_a-1){
				// Last Element
				nOfCols++;
				csc_col_a[nOfCols] = l+1;
			}
		} 
    
	
		//cout<<nOfCols<<endl;

		/*
		for(int i =0 ; i<=100;i++)
			cout<<csc_col_a[i]<<endl;

		for(int i=0;i<100;i++)
		   cout<<"Row:"<<csc_row_a[i]<<"\tval:"<<csc_val_a[i]<<endl;
	
	*/

		ifstream infile_b;
		infile_b.open(filenameB[f]);
		//infile_b.open("sM1000b.mtx");
		//infile_b.open("test2.txt");
		infile_b >> name; infile_b >> name; infile_b >> name ; infile_b >> name; infile_b >> name;
		infile_b >> nrows_b; infile_b >> ncols_b; infile_b >> nz_b;
		//printf("Filename B: %s, %d, %d, %d\n",filenameB[f],nrows_b,ncols_b,nz_b);
		cout<<"Filename:"<<filenameB[f]<<"\tRows:"<<nrows_b<<"\tColumns:"<<ncols_b<<"\tNon Zeros:"<<nz_b<<endl;

		R_B = (int *) malloc(nz_b * sizeof(int));
		C_B = (int *) malloc(nz_b * sizeof(int));
		V_B = (float *) malloc(nz_b * sizeof(float));  

		std::vector<std::tuple<int,int,float>> vectorB;

		for (int i=0; i<nz_b; i++){
			infile_b >> R_B[i]>>C_B[i]>>V_B[i];
			vectorB.push_back(std::make_tuple(C_B[i],R_B[i],V_B[i]));
		}
	
		infile_b.close();
		std::sort(begin(vectorB),end(vectorB));

		// Testing purpose only 
	/*	for(int i=0;i<vectorB.size();i++){
			cout<<"i:"<<i<<"\t"<< get<0>(vectorB[i])<<"\t"<<get<1>(vectorB[i])<<"\t"<<get<2>(vectorB[i])<<endl;		
		}
		*/
		// Create 2D - Matrix	
		// 1 Dim Array
		
		std::vector<float>csc_parallel_out;
		std::vector<float>csc_serial_out;

		float *csc_b = new float [nrows_b]; 
		float *csc_y = new float [nrows_b]; // Total number of NZ

	
	
		// Initializing B single array as zero
		for (int i = 0; i<nrows_b; i++){		
				csc_b[i]= 0;	
				csc_y[i]=0;
			}
	
		int m=0; // The size of col , val array
		int nc = nOfCols + 1;

		// Initializing B single array as zero
		for (int i = 0; i<nrows_b; i++){		
				csc_b[i]= 0;	
				csc_y[i]=0;
		}

		
		/***************************************************************/
		// sequential CSS...
		/***************************************************************/
		cout<<"Calculating...Sequential Time for CSC"<<endl;
		t = clock();
		for (int i=0, k=0; i<nz_b; i++)
		{				
			if (((i>0)&& (get<0>(vectorB[i]) - get<0>(vectorB[i-1]))!=0)){
				// new column in Matrix B started.
				k++;				
			
	
				for (int i=0; i<nOfCols; ++i) {
					csc_y[i] = 0.0;
					for (int j=csc_col_a[i]; j<csc_col_a[i+1]; ++j)
						csc_y[i] += csc_val_a[j]*csc_b[csc_row_a[j]];
				}

				// copy the results.
				std::copy(csc_y,csc_y+nrows_a,std::back_inserter(csc_serial_out));
			
				// Initialize a vector 			
				for (int l = 0; l<nrows_a; l++){		
					csc_b[l]= 0;		
					csc_y[l]=0;
				}
			}
			// pass the row data to the vector
			csc_b[get<1>(vectorB[i])-1] = get<2>(vectorB[i]);	

			if(i == (nz_b-1)) // last iteration
			{		
				for (int i=0; i<nOfCols; ++i) {
					csc_y[i] = 0.0;
					for (int j=csc_col_a[i]; j<csc_col_a[i+1]; ++j)
						csc_y[i] += csc_val_a[j]*csc_b[csc_row_a[j]];
				}

				// copy the results.
				std::copy(csc_y,csc_y+nrows_a,std::back_inserter(csc_serial_out));
			}

		}

		// End of Clock
		t = clock() - t;
		cout<<"CSC Sequential time in seconds:"<<((float)t)/CLOCKS_PER_SEC<<endl;

		/*for(int i=0;i<=99;i++){
			cout<<"i:\t"<<i<<"\t Serial:"<<csc_serial_out[i]<<"\t Parallel:"<<csc_parallel_out[i]<<endl;
		}*/
	
		m = 0; // Maximum array size
		/*****************************************************************/
		// Parallel 
		/*****************************************************************/
	
		cout<<"Calculating...Parallel Time for CSC: Filename:"<<filenameA[f]<<" && "<<filenameB[f]<<endl;
		t = clock();
		for (int i=0, k=0; i<nz_b; i++)
		{				
			if (((i>0)&& (get<0>(vectorB[i]) - get<0>(vectorB[i-1]))!=0)){
				// new column in Matrix B started.
				k++;				
				
				// pass the previous data to CUDA i.e csc_b
				// Multiply Matrix A with each Vector of B

				/*for(int p=0;p<20;p++){
					cout<<"csc_b:"<<csc_b[p]<<"csc_b llast :"<<csc_b[nrows_a-1-p]<<endl;
					
				}

				system("pause");
				*/
				cudaError_t cudaStatus = spmv_csc_matrix_multi(nc,csc_row_a, csc_col_a, 
					csc_val_a,csc_b, csc_y,nz_a,nrows_a,k);

				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "SPMV_CSC failed!");

					return 1;
				}		
				// copy the results.
				std::copy(csc_y,csc_y+nrows_a,std::back_inserter(csc_parallel_out));
			
				// Initialize a vector 			
				for (int l = 0; l<nrows_a; l++){		
					csc_b[l]= 0;		
					csc_y[l]=0;
				}
			}
			// pass the row data to the vector
			csc_b[get<1>(vectorB[i])-1] = get<2>(vectorB[i]);	

			if(i == (nz_b-1)) // last iteration
			{		
				// pass the previous data to CUDA i.e csc_b
				// Multiply Matrix A with each Vector of B
				cudaError_t cudaStatus = spmv_csc_matrix_multi(nc,csc_row_a, csc_col_a, 
						csc_val_a,csc_b, csc_y,nz_a,nrows_a, k);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "SPMV_CSC failed!");
					return 1;
				}		

				// copy the results.
				std::copy(csc_y,csc_y+nrows_a,std::back_inserter(csc_parallel_out));
			}

		}
	
		// End of Clock
		t = clock() - t;
		cout<<"CSC Parallel time in seconds:"<<((float)t)/CLOCKS_PER_SEC<<endl;
		cout<<"*-----------------------------------------------------------*"<<endl;		
	}
	system("pause");
	return 0;

}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t spmv_csc_matrix_multi( int h_nOfCols, int *h_csc_row_a, int *h_csc_col_a, float *h_csc_val_a, 
								   float *h_csc_b, float *h_y, int nz, int nrows,int k)
{
	float *dev_csc_val_a = 0;
    float *dev_csc_b = 0;
    int *dev_csc_col_a = 0;
	int *dev_csc_row_a = 0;
	float *dev_y = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .

		cudaStatus = cudaMalloc((void**)&dev_csc_val_a, nz * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&dev_csc_row_a, nz * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}


		cudaStatus = cudaMalloc((void**)&dev_csc_col_a, h_nOfCols * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

	cudaStatus = cudaMalloc((void**)&dev_csc_b, nrows * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_y, nrows * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_csc_row_a, h_csc_row_a, nz * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_csc_val_a, h_csc_val_a, nz * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_csc_col_a, h_csc_col_a, h_nOfCols * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

	cudaStatus = cudaMemcpy(dev_csc_b, h_csc_b, nrows * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_y, h_y, nrows * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


	int blockSize=1024;
	int gridSize = (h_nOfCols + blockSize - 1) / blockSize;

	/***************************************************************/
	// Launch a kernel on the GPU with one thread for each element.
	/**************************************************************/
    spmv_csc_matrix_kernel<<<gridSize,blockSize>>>(dev_csc_row_a, dev_csc_val_a,dev_csc_col_a, 
								  dev_csc_b, dev_y,nrows);
	
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_y, dev_y, nrows * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	   cudaStatus = cudaMemcpy(h_csc_b, dev_csc_b, nrows * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(h_csc_col_a, dev_csc_col_a, h_nOfCols * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(h_csc_val_a, dev_csc_val_a, nz * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

		cudaStatus = cudaMemcpy(h_csc_row_a, dev_csc_row_a, nz * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

/*
	for(int i=0;i<nrows;i++){
		cout<<"h_y\t"<<h_y[i]<<"\ti:\t"<<i<<endl;
	}
	*/

Error:
    cudaFree(dev_csc_b);
    cudaFree(dev_y);
	cudaFree(dev_csc_col_a);
	cudaFree(dev_csc_val_a);
	cudaFree(dev_csc_row_a);

    return cudaStatus;
}
