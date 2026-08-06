import torch
import triton
import triton.language as tl


@triton.jit
def build_A_kernel(
    K_ptr, ALPHA_ptr, BETA_ptr, A_ptr,
    C: tl.constexpr, D: tl.constexpr,
):
    offs_c=tl.arange(0, C)#creates a 1d tensor from 0 to chunk_length-1
    offs_d=tl.arange(0, D)

    #setting pointers to memory locations to load the data, offs_c[;,None] represents C,1. Final results:(C, D) grid of pointers, one for each element of k
    k_ptrs=K_ptr+offs_c[:, None]*D+offs_d[None, :]
    a_ptrs=ALPHA_ptr+offs_c[:, None]*D+offs_d[None, :]
    beta_ptrs=BETA_ptr+offs_c# only (C,)

    k = tl.load(k_ptrs)#loads tensor data from GPU AT address above
    alpha = tl.load(a_ptrs)             
    beta = tl.load(beta_ptrs)           

    log_alpha=tl.log(alpha)
    log_gamma=tl.cumsum(log_alpha, axis=0)#cumsum

    A = tl.zeros((C, C), dtype=tl.float32)
    for r in range(C):
        lg_r =tl.sum(tl.where(offs_c[:,None] == r,log_gamma, 0.0), axis=0)#Extract r from log_gamma, r is a row. tl.where masks all rows except r
        rel =tl.exp(lg_r[None, :]-log_gamma)#decay is computed for all i
        k_r = tl.sum(tl.where(offs_c[:, None] == r, k, 0.0), axis=0)#kr is extracted with shape D,
        row =tl.sum(k_r[None, :] * rel * k, axis=1)#A[r,i] before beta scaling
        beta_r=tl.sum(tl.where(offs_c == r, beta, 0.0))#def beta
        row =tl.where(offs_c < r, row * beta_r, 0.0)#creating a lower triangular matrux
        A =A+ tl.where(offs_c[:, None] == r, row[None, :], 0.0)#after beta scaling

    a_out_ptrs=A_ptr+ offs_c[:, None]*C+offs_c[None, :]
    tl.store(a_out_ptrs, A)


def test_build_A():
    ref = torch.load("step1_reference.pt")
    k, alpha, beta, A_expected = ref["k"], ref["alpha"], ref["beta"], ref["A"]
    C, D = k.shape

    k_cuda = k.cuda().contiguous()
    alpha_cuda = alpha.cuda().contiguous()
    beta_cuda = beta.cuda().contiguous()
    A_out = torch.zeros(C, C, device="cuda", dtype=torch.float32)

    build_A_kernel[(1,)](k_cuda, alpha_cuda, beta_cuda, A_out, C=C, D=D)

    A_out_cpu = A_out.cpu()
    diff = (A_out_cpu - A_expected).abs().max().item()
    print("Triton A =\n", A_out_cpu)
    print("\nExpected A =\n", A_expected)
    print(f"\nMax diff: {diff}")
    print("Close:", torch.allclose(A_out_cpu, A_expected, atol=1e-4))


if __name__ == "__main__":
    test_build_A()
