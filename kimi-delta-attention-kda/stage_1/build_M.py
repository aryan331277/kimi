import torch
import triton
import triton.language as tl


@triton.jit
#build A plus M
def build_M_kernel(
    K_ptr, ALPHA_ptr, BETA_ptr, M_ptr,
    C: tl.constexpr, D: tl.constexpr,
):
    offs_c=tl.arange(0, C)
    offs_d=tl.arange(0, D)

    k_ptrs=K_ptr+offs_c[:,None]*D+offs_d[None,:]
    a_ptrs=ALPHA_ptr+offs_c[:, None]*D+offs_d[None,:]
    beta_ptrs=BETA_ptr+offs_c

    k = tl.load(k_ptrs)
    alpha = tl.load(a_ptrs)
    beta = tl.load(beta_ptrs)

    log_alpha = tl.log(alpha)
    log_gamma = tl.cumsum(log_alpha, axis=0)

    # ---- building A, same as above formatted properly with no comments
    A = tl.zeros((C, C), dtype=tl.float32)
    for r in range(C):
        lg_r = tl.sum(tl.where(offs_c[:, None] == r, log_gamma, 0.0), axis=0)
        rel = tl.exp(lg_r[None, :] - log_gamma)
        k_r = tl.sum(tl.where(offs_c[:, None] == r, k, 0.0), axis=0)
        row = tl.sum(k_r[None, :] * rel * k, axis=1)
        beta_r = tl.sum(tl.where(offs_c == r, beta, 0.0))
        row = tl.where(offs_c < r, row * beta_r, 0.0)
        A += tl.where(offs_c[:, None] == r, row[None, :], 0.0)

    I_full= tl.where(offs_c[:, None] == offs_c[None,:],1.0,0.0)#setting up the identity matrix
    L =I_full + A

    Minv= tl.zeros((C,C),dtype=tl.float32)#this is (I+A)^-1 but here beta col scale isnt done
    for c in range(C):#row by row forward sub
        acc=tl.zeros((C,), dtype=tl.float32)
        for l in range(c):
            a_cl=tl.sum(tl.where((offs_c == c)[:, None] & (offs_c == l)[None, :], L, 0.0))#loop over all previous rows, then acc the sum
            m_l=tl.sum(tl.where(offs_c[:, None] == l, Minv, 0.0), axis=0)#ext single element then add
            acc=acc+a_cl*m_l#extract and acc
        e_c=tl.where(offs_c == c, 1.0, 0.0)
        Minv=Minv+ tl.where(offs_c[:, None] == c, (e_c - acc)[None, :], 0.0)#scatter

    M = Minv * beta[None, :]#diagonal beta is applied to the right columns

    m_out_ptrs = M_ptr + offs_c[:, None] * C + offs_c[None, :]#def pointers
    tl.store(m_out_ptrs, M)


def test_build_M():
    ref = torch.load("step1_reference.pt")
    k, alpha, beta, M_expected = ref["k"], ref["alpha"], ref["beta"], ref["M"]
    C, D = k.shape

    k_cuda = k.cuda().contiguous()
    alpha_cuda = alpha.cuda().contiguous()
    beta_cuda = beta.cuda().contiguous()
    M_out = torch.zeros(C, C, device="cuda", dtype=torch.float32)

    build_M_kernel[(1,)](k_cuda, alpha_cuda, beta_cuda, M_out, C=C, D=D)

    M_out_cpu = M_out.cpu()
    diff = (M_out_cpu - M_expected).abs().max().item()
    print("Triton M =\n", M_out_cpu)
    print(f"Max diff: {diff}")
    print("Close:", torch.allclose(M_out_cpu, M_expected, atol=1e-4))


if __name__ == "__main__":
    test_build_M()
