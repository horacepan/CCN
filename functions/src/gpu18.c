#include <THC.h>
#include <THCGeneral.h>

#include "RisiContraction_18_gpu_cuda.h"

extern THCState *state;

int RiskContraction18_forward(
	THCudaTensor* F_tensor,
	THCudaTensor* adj_tensor,
	THCudaTensor* output_tensor,
	int N,
	int nChannels
) {
	RisiContraction_18_forward_kernel(
		state,
		F_tensor,
		adj_tensor,
		output_tensor,
		N,
		nChannels
	);

	return 1;
}

int RiskContraction18_backward(
	THCudaTensor *gradient_input_tensor, 
	THCudaTensor *adj_tensor, 
	THCudaTensor *gradient_output_tensor,
	int N,
	int nChannels
) {
	RisiContraction_18_backward_kernel(
		state,
		gradient_input_tensor,
		adj_tensor,
		gradient_input_tensor,
		N,
		nChannels
	);

	return 1;
}