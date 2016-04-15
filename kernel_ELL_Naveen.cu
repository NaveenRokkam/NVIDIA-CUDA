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

cudaError_t spmv_ell_matrix_multi( int num_rows,  int num_cols_per_row, 
								   int * indices,  float * data, float * x, float *y,int m );

__global__ void spmv_ell_matrix_kernel ( const int num_rows, const int num_cols_per_row , 
				  const int * indices, const float * data, const float * x , float * y)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x ;
	if( row < num_rows ){
		float dot = 0;
		for ( int n = 0; n < num_cols_per_row ; n ++){
			int col = indices [ num_rows * n + row ];
			float val = data [ num_rows * n + row ];
			if( val != 0)
				dot += val * x [ col ];
		}
		y[ row ] += dot ;
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
		//infile.open("sM1000a.mtx");
		infile.open(filenameA[f]);	
		infile >> name; infile >> name; infile >> name ; infile >> name; infile >> name;
		infile >> nrows_a; infile >> ncols_a; infile >> nz_a;
		cout<<"Filename:"<<filenameA[f]<<"\tRows:"<<nrows_a<<"\tColumns:"<<ncols_a<<"\tNon Zeros:"<<nz_a<<endl;

		R_A = (int *) malloc(nz_a * sizeof(int));
		C_A = (int *) malloc(nz_a * sizeof(int));
		V_A = (float *) malloc(nz_a * sizeof(float));
    
		int row_check = 0;
		int len = 0; int len_row_a = 0;
		for (int i=0; i<nz_a; i++)
		{
			infile >> R_A[i]>>C_A[i]>>V_A[i];
       
			// check how many values in the current row
			if(row_check == R_A[i]){
				len++;  // Number of values in a row
			}
			else{ 
				row_check = R_A[i];  // Take the new row for comparison 
				if (len_row_a < len ){
					len_row_a = len;	// Maximum length of a row in the ELL Matrix			
				}
				len = 0;			
			} 		
		}
	
		ifstream infile_b;
		//infile_b.open("sM1000b.mtx");
		infile_b.open(filenameB[f]);
		infile_b >> name; infile_b >> name; infile_b >> name ; infile_b >> name; infile_b >> name;
		infile_b >> nrows_b; infile_b >> ncols_b; infile_b >> nz_b;
		//printf("%d, %d, %d\n",nrows_b,ncols_b,nz_b);
		cout<<"Filename:"<<filenameB[f]<<"\tRows:"<<nrows_b<<"\tColumns:"<<ncols_b<<"\tNon Zeros:"<<nz_b<<endl;

		R_B = (int *) malloc(nz_b * sizeof(int));
		C_B = (int *) malloc(nz_b * sizeof(int));
		V_B = (float *) malloc(nz_b * sizeof(float));  

		std::vector<std::tuple<int,int,float>> vectorB;

		for (int i=0; i<nz_b; i++){
			infile_b >> R_B[i]>>C_B[i]>>V_B[i];
			vectorB.push_back(std::make_tuple(C_B[i],R_B[i],V_B[i]));
		}
	
		std::sort(begin(vectorB),end(vectorB));

		// Testing purpose only 
	/*	for(int i=0;i<vectorB.size();i++){
			cout<<"i:"<<i<<"\t"<< get<0>(vectorB[i])<<"\t"<<get<1>(vectorB[i])<<"\t"<<get<2>(vectorB[i])<<endl;		
		}
		*/
		// Create 2D - Matrix	
		// 1 Dim Array
		int **ell_col_a = new int *[nrows_a];
		float **ell_val_a = new float *[nrows_a];
		float *ell_b = new float [nrows_b];	
		std::vector<float>ell_parallel_out;
		std::vector<float>ell_serial_out;
		//float *ell_parallel_out = new float[nrows_a * ncols_b];
		//float *ell_serial_out = new float[nrows_a * ncols_b];
	
		// Create 2D Array
		for (int i = 0; i<nrows_a; i++){
			ell_col_a[i] = new int[len_row_a];
			ell_val_a[i] = new float[len_row_a];
		}

		// Initialize the array to Zero to handle * values
		for (int i = 0; i<nrows_a; i++){
			for(int j = 0; j<=len_row_a;j++){
				ell_col_a[i][j]= 0;	
				ell_val_a[i][j]= 0;
			}}

		for (int i = 0; i<nrows_b; i++){		
				ell_b[i]= 0;	
			}
	
		// Fill Values
		int j=0;
		for (int i=0; i<nz_a; i++)
		{		
			if((i>1)&&((R_A[i] - R_A[i-1]) != 0))
				j=0;  // Start new column

			ell_col_a[R_A[i]-1][j] = C_A[i]; // 
		//	cout<< "ell_col[R[i]-1][j]: "<<ell_col[R[i]-1][j]<<"C[i]:"<<C[i]<<endl;
			ell_val_a[R_A[i]-1][j] = V_A[i];
			j++; // Increment the coloum
		}

		int m=0; // The size of col , val array
		int *col_a = new int[(nrows_a*(len_row_a+1))]; // a single array for holding all values
		float *val_a = new float[(nrows_a*(len_row_a+1))];
		for(int c=0;c<=len_row_a;c++){
			for (int r=0;r<nrows_a;r++){
				if(m>=(nrows_a*(len_row_a+1))){
					goto next;
				}
				if ((ell_col_a[r][c]) > 0)
					col_a[m]= ell_col_a[r][c]-1; // Columns start with 0 index hence	
				else
					col_a[m]= ell_col_a[r][c];
				val_a[m]= ell_val_a[r][c];
		//		cout<<"r:"<<r<<"c:"<<c<<"\t"<<"Col:"<<col_a[m]<<"\t"<<"val:"<<val_a[m]<<"\t M:"<<m<<endl;
				m++;
			}
		}

		next:
		//system("pause");

		/*****************************************************************/
		// Parallel 
		/*****************************************************************/
		cout<<"Calculating...Parallel Time for CSC: Filename:"<<filenameA[f]<<" && "<<filenameB[f]<<endl;
		t = clock();

		float *ell_y = new float [nrows_a];	
		for(int i =0; i<nrows_a;i++) ell_y[i]=0;

		for (int i=0, j=0; i<nz_b; i++)
		{				
			if (((i>0)&& (get<0>(vectorB[i]) - get<0>(vectorB[i-1]))!=0)){
				// new column started.
				j++;				

				// pass the previous data to CUDA i.e ell_b
				// Multiply Matrix A with each Vector of B
				cudaError_t cudaStatus = spmv_ell_matrix_multi(nrows_a, (len_row_a+1),
											col_a, val_a,ell_b, ell_y,m);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "SPMV_ELL failed!");
					return 1;
				}		
			
				// copy the results.
				std::copy(ell_y,ell_y+nrows_a,std::back_inserter(ell_parallel_out));
			
				// Initialize a vector 			
					for (int l = 0; l<nrows_b; l++){		
					ell_b[l]= 0;		
					ell_y[l]=0;
				}
			
	

			}
			// pass the row data to the vector
			ell_b[get<1>(vectorB[i])-1] = get<2>(vectorB[i]);	

			if(i == (nz_b-1)) // last iteration
			{
				// pass the previous data to CUDA i.e ell_b
				// Multiply Matrix A with each Vector of B
				cudaError_t cudaStatus = spmv_ell_matrix_multi(nrows_a, (len_row_a+1),
											col_a, val_a,ell_b, ell_y,m);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "SPMV_ELL failed!");
					return 1;
				}		
			
				// copy the results.
				std::copy(ell_y,ell_y+nrows_a,std::back_inserter(ell_parallel_out));
			}

		}
		// End of Clock
		t = clock() - t;
		cout<<"CSC Parallel time in seconds:"<<((float)t)/CLOCKS_PER_SEC<<endl;
		
	

		// Final Parallel Output available in vector: ell_parallel_out.

		m = 0; // Maximum array size
		
		/***************************************************************/
		// sequential ELL...
		/***************************************************************/
		
		cout<<"Calculating...Sequential Time for CSC"<<endl;
		t = clock();
		for(int i =0; i<nrows_a;i++) { 
			ell_y[i]=0; ell_b[i]=0;
		}

		for(int r=0;r<nrows_a;r++){
			for (int c=0;c<=len_row_a;c++){
				if(m>=(nrows_a*(len_row_a+1))){
					goto serial;
				}
				if ((ell_col_a[r][c]) > 0)
					col_a[m]= ell_col_a[r][c]-1; // Columns start with 0 index hence	
				else
					col_a[m]= ell_col_a[r][c];
				val_a[m]= ell_val_a[r][c];
		//		cout<<"r:"<<r<<"c:"<<c<<"\t"<<"Col:"<<col_a[m]<<"\t"<<"val:"<<val_a[m]<<"\t M:"<<m<<endl;
				m++;
			}
		}

		serial:

		for (int i=0, k=0; i<nz_b; i++)
		{				
			if (((i>0)&& (get<0>(vectorB[i]) - get<0>(vectorB[i-1]))!=0)){
				// new column started.
				k++;				
			
				int jj = 0; int c;
				for(int i=0; i<nrows_a;i++){
					ell_y[i]=0;
					for (int j=0; j<=len_row_a; j++) {
						jj = j + (len_row_a+1)*i;
						c = col_a[jj];
						if ((c >= 0) && (c < nrows_a))
							ell_y[i] += val_a[jj] * ell_b[c];
					}

				}

				// copy the results.
				std::copy(ell_y,ell_y+nrows_a,std::back_inserter(ell_serial_out));
			
				// Initialize a vector 			
				for (int l = 0; l<nrows_a; l++){		
					ell_b[l]= 0;		
					ell_y[l]=0;
				}
			}
			// pass the row data to the vector
			ell_b[get<1>(vectorB[i])-1] = get<2>(vectorB[i]);	

			if(i == (nz_b-1)) // last iteration
			{		
				int jj = 0; int c;
				for(int i=0; i<nrows_a;i++){
					ell_y[i]=0;
					for (int j=0; j<=len_row_a; j++) {
						jj = j + (len_row_a+1)*i;
						c = col_a[jj];
						if ((c >= 0) && (c < nrows_a))
							ell_y[i] += val_a[jj] * ell_b[c];
					}

				}

				// copy the results.
				std::copy(ell_y,ell_y+nrows_a,std::back_inserter(ell_serial_out));
			}

		}
		// End of Clock
		t = clock() - t;
		cout<<"CSC Sequential time in seconds:"<<((float)t)/CLOCKS_PER_SEC<<endl;
		cout<<"*-----------------------------------------------------------*"<<endl;	

/*		//for(int i=0;i<100;i++){
		for(int i=0;i<(nrows_a*ncols_a);i++){
			cout<<"i:\t"<<i<<"\t Serial:"<<ell_serial_out[i]<<"\t Parallel:"<<ell_parallel_out[i]<<endl;
		}*/
	}
	system("pause");
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t spmv_ell_matrix_multi( int h_num_rows, int h_num_cols_per_row, 
								  int *h_indices, float *h_data, float *h_x, float *h_y, int m )
{
	float *dev_x = 0;
    float *dev_y = 0;
    int *dev_indices = 0;
	float *dev_data = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    
    cudaStatus = cudaMalloc((void**)&dev_y, h_num_rows * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_x, h_num_rows * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_indices, m * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_data, m * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_x, h_x, h_num_rows * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_y, h_y, h_num_rows * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_indices, h_indices, m * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_data, h_data, m * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	int blockSize=1024;
	int gridSize = (h_num_rows + blockSize - 1) / blockSize; 


	/**************************************************************/
	// Launch a kernel on the GPU with one thread for each element.
	/**************************************************************/
    spmv_ell_matrix_kernel<<<gridSize,blockSize>>>(h_num_rows, h_num_cols_per_row,dev_indices, 
								  dev_data, dev_x , dev_y);
	
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
    cudaStatus = cudaMemcpy(h_x, dev_x, h_num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	   cudaStatus = cudaMemcpy(h_y, dev_y, h_num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(h_indices, dev_indices, m * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(h_data, dev_data, m * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

/*
	for(int i=0;i<h_num_rows;i++){
		cout<<"h_y\t"<<h_y[i]<<"\ti:\t"<<i<<endl;
	}
*/

Error:
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_indices);
	cudaFree(dev_data);
    
    return cudaStatus;
}
