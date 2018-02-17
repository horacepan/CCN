#ifdef __cplusplus
	extern "C" {
#endif

void RisiContraction_18_forward_kernel(
	THCState* state,
	THCudaTensor* F_tensor,
	THCudaTensor* adj_tensor,
	THCudaTensor* output_tensor,
	int N,
	int nChannels
);

void RisiContraction_18_backward_kernel(
	THCState* state,
	THCudaTensor* result_tensor,
	THCudaTensor* adj_tensor,
	THCudaTensor* gradient_tensor,
	int N,
	int nChannels
);

#ifdef __cplusplus
	}
#endif
