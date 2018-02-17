// Framework: GraphFlow
// Class: RisiContraction_18_gpu
// Author: Machine Learning Group of UChicago
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#include <THC.h>
#include <THCGeneral.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "RisiContraction_18_gpu_cuda.h"

#ifdef __cplusplus
	extern "C" {
#endif

#define NUM_THREADS 512
#define BLOCK 512
#define NUM_CONTRACTIONS 18

__global__ void RisiContraction_18_forward_job(float *tensor, float *adj, float *value, int N, int nChanels) {
	
	__shared__ int nContractions;
	__shared__ int A;
	__shared__ int B;
	__shared__ int C;
	__shared__ int Y;


	nContractions = NUM_CONTRACTIONS;

	int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (global_threadId < N * N * nChanels * nContractions) {	
		C = nChanels;
		B = N * C;
		A = N * B;

		Y = nChanels * nContractions;
		
		int f = (global_threadId % Y) % nChanels;
		int Case = (global_threadId % Y) / nChanels + 1;
		int y = (global_threadId / Y) % N;
		int x = (global_threadId / Y) / N;

		int a, b, c, d, e;
		float adj_value;

		float sum = 0.0;

		// +-----------+
		// | 1 + 1 + 1 |
		// +-----------+
		
		// Case 1 (1/50): Fix a, b. Contract c, d, e.
		if (Case == 1) {
			a = x;
			b = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					if (adj_value > 0) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}
		}

				
		// Case 2 (3/50): Fix a, d. Contract b, c, e.
		if (Case == 2) {		
			a = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (b = 0; b < N; ++b) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}	
		}
		
		// Case 3 (5/50): Fix b, c. Contract a, d, e.
		if (Case == 3) {		
			b = x;
			c = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					if (adj_value > 0) {
						for (a = 0; a < N; ++a) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}	
		}

		// Case 4 (6/50): Fix b, d. Contract a, c, e.
		if (Case == 4) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (a = 0; a < N; ++a) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}
		}

		// Case 5 (10/50): Fix d, e. Contract a, b, c.
		if (Case == 5) {		
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (a = 0; a < N; ++a) {
					for (b = 0; b < N; ++b) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}
		}

		// +-------+
		// | 1 + 2 |
		// +-------+

		// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
		if (Case == 6) {
			a = x;
			b = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					c = d;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}

		// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
		if (Case == 7) {
			a = x;
			b = y;

			for (d = 0; d < N; ++d) {
				e = d;
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
		if (Case == 8) {
			a = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (b = 0; b < N; ++b) {
						c = b;
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
		if (Case == 9) {
			a = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					b = e;
					for (c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
		if (Case == 10) {
			b = x;
			c = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					if (adj_value > 0) {
						a = d;
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
		if (Case == 11) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (a = 0; a < N; ++a) {
						c = a;
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
		if (Case == 12) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					a = e;
					for (int c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
		if (Case == 13) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					c = e;
					for (int a = 0; a < N; ++a) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
		if (Case == 14) {
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (int a = 0; a < N; ++a) {
					b = a;
					for (int c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
		if (Case == 15) {
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (int b = 0; b < N; ++b) {
					c = b;
					for (int a = 0; a < N; ++a) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// +---+
		// | 3 |
		// +---+

		// Case 16 (43/50): (a, d). Contract (b, c, e).
		if (Case == 16) {
			a = x;
			d = y;

			for (int e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					b = e;
					c = e;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}	

		// Case 17 (46/50): (b, d). Contract (a, c, e).
		if (Case == 17) {
			b = x;
			d = y;

			for (int e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					a = e;
					c = e;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}

		// Case 18 (50/50): (d, e). Contract (a, b, c).
		if (Case == 18) {
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (int a = 0; a < N; ++a) {
					b = a;
					c = a;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}
		value[global_threadId] = sum;
	}
}


__global__ void RisiContraction_18_backward_job(float *tensor_gradient, float *adj, float *gradient, int N, int nChanels) {
	
	__shared__ int nContractions;
	__shared__ int X;
	__shared__ int Y;

	nContractions = NUM_CONTRACTIONS;

	int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_threadId < N * N * N * nChanels) {
		X = N * nChanels * nContractions;
		Y = nChanels * nContractions;

		int f = global_threadId % nChanels;
		int c = (global_threadId / nChanels) % N;
		int b = ((global_threadId / nChanels) / N) % N;
		int a = ((global_threadId / nChanels) / N) / N;

		float sum = 0.0;

		int ind;
		float adj_value;

		for (int d = 0; d < N; ++d) {
			for (int e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];

				if (adj_value > 0) {
					// +-----------+
					// | 1 + 1 + 1 |
					// +-----------+

					// Case 1 (1/50): Fix a, b. Contract c, d, e.
					ind = a * X + b * Y + 0 * nChanels + f;
					sum += gradient[ind] * adj_value;


					// Case 2 (3/50): Fix a, d. Contract b, c, e.
					ind = a * X + d * Y + 1 * nChanels + f;
					sum += gradient[ind] * adj_value;

					// Case 3 (5/50): Fix b, c. Contract a, d, e.
					ind = b * X + c * Y + 2 * nChanels + f;
					sum += gradient[ind] * adj_value;

					// Case 4 (6/50): Fix b, d. Contract a, c, e.
					ind = b * X + d * Y + 3 * nChanels + f;
					sum += gradient[ind] * adj_value;

					// Case 5 (10/50): Fix d, e. Contract a, b, c.
					ind = d * X + e * Y + 4 * nChanels + f;
					sum += gradient[ind] * adj_value;

					// +-------+
					// | 1 + 2 |
					// +-------+

					// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
					if (c == d) {
						ind = a * X + b * Y + 5 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
					if (d == e) {
						ind = a * X + b * Y + 6 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
					if (b == c) {
						ind = a * X + d * Y + 7 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
					if (b == e) {
						ind = a * X + d * Y + 8 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
					if (a == d) {
						ind = b * X + c * Y + 9 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
					if (a == c) {
						ind = b * X + d * Y + 10 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
					if (a == e) {
						ind = b * X + d * Y + 11 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
					if (c == e) {
						ind = b * X + d * Y + 12 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
					if (a == b) {
						ind = d * X + e * Y + 13 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
					if (b == c) {
						ind = d * X + e * Y + 14 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// +---+
					// | 3 |
					// +---+

					// Case 16 (43/50): (a, d). Contract (b, c, e).
					if ((b == c) && (c == e))  {
						ind = a * X + d * Y + 15 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 17 (46/50): (b, d). Contract (a, c, e).
					if ((a == c) && (c == e))  {
						ind = b * X + d * Y + 16 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 18 (50/50): (d, e). Contract (a, b, c).
					if ((a == b) && (b == c))  {
						ind = d * X + e * Y + 17 * nChanels + f;
						sum += gradient[ind] * adj_value;
					}
				}
			}
		}
		
		tensor_gradient[global_threadId] += sum;
	}
}


int rounded_division(int number1, int number2) {
	if (number1 % number2 == 0) {
		return number1 / number2;
	}
	return number1 / number2 + 1;
}

dim3 cuda_gridsize(int n){
	int k = (n - 1) / BLOCK + 1;
	int x = k;
	int y = 1;
	if (x > 65535){
		x = ceil(sqrt(k));
		y = (n - 1) / (x * BLOCK) + 1;
	}
	dim3 d(x, y, 1);
	return d;
}
		

void RisiContraction_18_forward_kernel(
	THCState* state,
	THCudaTensor* F_tensor,
	THCudaTensor* adj_tensor,
	THCudaTensor* output_tensor,
	int N,
	int nChannels
){
	
	float* F = THCudaTensor_data(state, F_tensor);
	float* adj = THCudaTensor_data(state, adj_tensor);
	float* output = THCudaTensor_data(state, output_tensor);
	cudaStream_t stream = THCState_getCurrentStream(state);
	
	int size = N * N * nChannels * NUM_CONTRACTIONS;
	int nThreads = NUM_THREADS;
	dim3 dimGrid(rounded_division(size, nThreads));
	dim3 dimBlock(nThreads);
	cudaError_t err;

	//RisiContraction_18_forward_job<<<dimGrid, dimBlock, 0, stream>>>(F, adj, output, N, nChannels);
	RisiContraction_18_forward_job<<<cuda_gridsize(size), BLOCK, 0, stream>>>(F, adj, output, N, nChannels);

	err = cudaGetLastError();
	if (cudaSuccess != err){
		fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}


void RisiContraction_18_backward_kernel(
	THCState* state,
	THCudaTensor* result_tensor,
	THCudaTensor* adj_tensor,
	THCudaTensor* gradient_tensor,
	int N,
	int nChannels
){
	
	float* gradient = THCudaTensor_data(state, gradient_tensor);
	float* adj = THCudaTensor_data(state, adj_tensor);
	float* result = THCudaTensor_data(state, result_tensor);
	cudaStream_t stream = THCState_getCurrentStream(state);

	int size = N * N * nChannels;
	int nThreads = NUM_THREADS;
	dim3 dimGrid(rounded_division(size, nThreads));
	dim3 dimBlock(nThreads);
	cudaError_t err;

	//RisiContraction_18_backward_job<<<dimGrid, dimBlock, 0, stream>>>(result, adj, gradient, N, nChannels);
	RisiContraction_18_backward_job<<<cuda_gridsize(size), BLOCK, 0, stream>>>(result, adj, gradient, N, nChannels);

	err = cudaGetLastError();
	if (cudaSuccess != err){
		fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

#ifdef __cplusplus
	}
#endif
