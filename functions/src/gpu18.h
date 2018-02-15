int RiskContraction18_forward(
	THCudaTensor *F_tensor, 
	THCudaTensor *adj_tensor, 
	THCudaTensor *output_tensor,
	int N,
	int nChannels);

int RiskContraction18_backward(
	THCudaTensor *gradient_input_tensor, 
	THCudaTensor *adj_tensor, 
	THCudaTensor *gradient_output_tensor,
	int N,
	int nChannels);