#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cmath>
#include <iostream>

#define Br 32
#define Bc 32
#define d 64
/*
ToDo:
1. Debug done, implement atomicmaxFLoat
2. test further
*/
__device__ __forceinline__ 
float atomicMaxFloat(float* addr, float val)
{
   // this is ret value where we store the value that the pointer (addr) is pointing to and converts to int
   int ret = __float_as_int(*addr);

   // while the value is bigger than the float(ret) this is done to check race conditions and to truly atomicize it as if another thread is parallely also checking it and a switch was made then that would make the CAS wrong.
   while(val > __int_as_float(ret))
   {
        int old = ret;
        // atomicCAS(*addr,compare,value) deref the address pointer to find the value 'old'
        // CAS mathwise is doing -> (old == compare ? val : old) on the address pointer. which says if old is equal to compare then old is overwritten to value, else nothing

        if((ret = atomicCAS((int*)addr,old,__float_as_int(val))) == old)
            break; // this statement is checking after the CAS operation is done whether the memory value at ret equals to old, if yes then swap is successful, otherwise other thread changed value
   }
   return __int_as_float(ret);
}

__global__
void fwd_pass_kernel(float* Q,
                    float* V, 
                    float* K, 
                    int N,
                    int Tc, 
                    int Tr, 
                    float softmax_scale, 
                    float* O)
{

    //shared memory which is used to chunk the big NxN matrix and for it to never materialize.
    __shared__ float Q_i[Br][d];
    __shared__ float K_j[Bc][d];
    __shared__ float V_j[Bc][d];
    
    // the chunks above are required for HBM -> SRAM and vice versa
    
    __shared__ float S_i[Bc][Br]; // Just scaled mat-mul
    __shared__ float P_i[Bc][Br]; // activated probability (will check later if actually needed)
    __shared__ float O_i[Br][d]; // this is required for SRAM -> HBM for final o/p
    __shared__ float m_i[Br]; // For max value tracking in softmax
    __shared__ float l_i[Br]; // For sum of exp values in softmax

    // outer for loop of rows(change made in flashattention 2)
    // THINK IN BLOCKS!! i here is the representation of BLOCKS not of rows.
    for(int i=0; i<Tr; i++) 
    {
        // load Q_i tile, from HBM TO SRAM:
        for(int row = threadIdx.y; row<Br;row+=blockDim.y)
        {
            for(int col = threadIdx.x; col<Bc;col+=blockDim.x)
            {
                int global_row = i*Br + row;
                Q_i[row][col] = Q[global_row*d + col]; 
            }
        }
        __syncthreads();
        // Initialize O_i to zero
        for (int row = threadIdx.y; row < Br; row += blockDim.y) {
            for (int col = threadIdx.x; col < d; col += blockDim.x) {
                O_i[row][col] = 0.0f;  // O_i <- 0
            }
        }
        // Initialize m_i and l_i
        for (int row = threadIdx.y; row < Br; row += blockDim.y) {
            int global_row = i * Br + row;
            m_i[global_row] = -INFINITY;  // m_i <- -inf
            l_i[global_row] = 0.0f;       // l_i <- 0
        }
        __syncthreads();

        // inner loop where all the compute actually happens.
        for(int j=0 ; j<Tc ; j++)
        {
            // loading K_j and V_j
            for(int row = threadIdx.y; row<Bc;row+=blockDim.y)
            {
                for(int col = threadIdx.x; col<d;col+=blockDim.x)
                {
                    int global_row = j*Bc + row;
                    K_j[row][col] = K[global_row*d + col];
                    V_j[row][col] = V[global_row*d + col]; 
                }
            }
        __syncthreads();
            // S_i = Q_i[Br , d] @ [d, Bc]K_j.T 
            for(int row = threadIdx.y; row < Br; row += blockDim.y)
            {
                for(int col = threadIdx.x; col < Bc; col += blockDim.x)
                {
                    float acc = 0; 
                    for(int k=0 ; k<d ; k++)
                    {
                        acc += Q_i[row][k] * K_j[k][col];
                    }
                    S_i[row][col] = softmax_scale * acc;
                }
            }
        __syncthreads();

            // tip: IF POSSIBLE just do it in one for loop, the row is a shared for loop for updating m, updating l and updating P
            for(int row = threadIdx.y; row < Br; row += blockDim.y)
            {  
                // update m_i
                float m_i_old = m_i[row];

                float row_max = -INFINITY;
                for(int col = threadIdx.x; col < Bc; col += blockDim.x)
                {
                    row_max = fmaxf(row_max,S_i[row][col]);
                }
                float m_i_new = fmaxf(m_i_old,row_max);
                atomicMaxFloat(&m_i[row],m_i_new);
        __syncthreads();

                // after updated maximum, calculate the shifted probability tile. Shifted because exponents have a tendency to explode.
                // Find new P_i and update row_sum for l_i 
                float row_sum = 0;
                for(int col = threadIdx.x; col < Bc; col += blockDim.x)
                {
                    P_i[row][col] = expf(S_i[row][col]-m_i[row]);
                    row_sum += P_i[row][col];
                }
                float l_i_new = 0.0f;

                if(m_i_old != -INFINITY) 
                {
                    l_i_new = l_i[row]*expf(m_i_old-m_i_new) + row_sum;
                }
                else{
                    l_i[row] = row_sum;
                }
        __syncthreads();
                // For o_i^1 as in the paper u need to handle the first case seperately, reason: (exp(-inf))
                if(m_i_old == -INFINITY) 
                {
                    for(int d_idx = 0; d_idx<d; d_idx++)
                    {
                        float sum = 0.0f;
                        for(int col = 0; col<Bc; col++)
                        {
                            sum += P_i[row][col]*V_j[col][d_idx];
                        }
                    }
                }
                // O_i^j with the exp scaling factor.
                else
                {
                    float scale = expf(m_i_new-m_i_old);
                    for(int d_idx = 0; d_idx < d; d_idx++) 
                    {
                        O_i[row][d_idx] *= scale;
                        
                        float sum = 0.0f;
                        for(int col = 0; col < Bc; col++) 
                        {
                            sum += P_i[row][col] * V_j[col][d_idx];
                        }
                        O_i[row][d_idx] += sum;
                    }
                }
            }
        }
        __syncthreads();
        // outside first for loop we have to add the final scaling factor (improvments on flash attention 1 as now we only rescale once at the end, saves on FLOPS)
        for (int row = threadIdx.y; row < Br; row += blockDim.y) 
        {
            for (int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x) 
            {
                O_i[row][d_idx] /= l_i[row];  // O_i = Õ_i / l_i
            }
        }
        __syncthreads();

        // Write O_i into O, SRAM -> HBM
        for(int row = threadIdx.y; row < Br; row += blockDim.y)
        {
            int global_row = i*Br + row; 
            for(int d_idx = threadIdx.x; d_idx < d; d_idx += blockDim.x)
            {
                O[global_row*d + d_idx] = O_i[row][d_idx];
            }
        }
    }
}


torch::Tensor fwd_pass(torch::Tensor Q,torch::Tensor K,torch::Tensor V)
{

    int N = Q.size(0);

    int Tr = N+Br-1/Br; 
    int Tc = N+Bc-1/Bc; 

    float softmax_scale = 1.0f/sqrtf(d);

    Q = Q.contiguous().to(torch::kCUDA);
    V = V.contiguous().to(torch::kCUDA);
    K = K.contiguous().to(torch::kCUDA);

    auto O = torch::zeros({N,d},Q.options());

    
    dim3 blockDim(Br,Bc);
    dim3 gridDim(Tr,Tc);

    fwd_pass_kernel<<<blockDim,gridDim>>>(
            Q.data_ptr<float>(),
            V.data_ptr<float>(),
            K.data_ptr<float>(),
            N,
            Tc,Tr,
            softmax_scale,
            O.data_ptr<float>()
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
{
    m.def("fwd_pass",&fwd_pass,"Flash Attention 2 forward pass (CUDA)");
}