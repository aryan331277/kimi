#combines both of the kernels, and adds WU implementation until now eqn 1-5 is done and implemented

import torch
import triton
import triton.language as tl


@triton.jit
def build_WU_kernel(
    K_ptr, V_ptr, ALPHA_ptr, BETA_ptr, W_ptr, U_ptr,
    C: tl.constexpr, D: tl.constexpr,
):
    offs_c = tl.arange(0, C)
    offs_d = tl.arange(0, D)

    k_ptrs = K_ptr + offs_c[:, None] * D + offs_d[None, :]
    v_ptrs = V_ptr + offs_c[:, None] * D + offs_d[None, :]
    a_ptrs = ALPHA_ptr + offs_c[:, None] * D + offs_d[None, :]
    beta_ptrs = BETA_ptr + offs_c

    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    alpha = tl.load(a_ptrs)
    beta = tl.load(beta_ptrs)

    log_alpha = tl.log(alpha)
    log_gamma = tl.cumsum(log_alpha, axis=0)

    A = tl.zeros((C, C), dtype=tl.float32)
    for r in range(C):
        lg_r = tl.sum(tl.where(offs_c[:, None] == r, log_gamma, 0.0), axis=0)
        rel = tl.exp(lg_r[None, :] - log_gamma)
        k_r = tl.sum(tl.where(offs_c[:, None] == r, k, 0.0), axis=0)
        row = tl.sum(k_r[None, :] * rel * k, axis=1)
        beta_r = tl.sum(tl.where(offs_c == r, beta, 0.0))
        row = tl.where(offs_c < r, row * beta_r, 0.0)
        A += tl.where(offs_c[:, None] == r, row[None, :], 0.0)

    I_full = tl.where(offs_c[:, None] == offs_c[None, :], 1.0, 0.0)
    L = I_full + A

    Minv = tl.zeros((C, C), dtype=tl.float32)
    for c in range(C):
        acc = tl.zeros((C,), dtype=tl.float32)
        for l in range(c):
            a_cl = tl.sum(tl.where((offs_c == c)[:, None] & (offs_c == l)[None, :], L, 0.0))
            m_l = tl.sum(tl.where(offs_c[:, None] == l, Minv, 0.0), axis=0)
            acc += a_cl * m_l
        e_c = tl.where(offs_c == c, 1.0, 0.0)
        Minv=Minv+ tl.where(offs_c[:, None] == c, (e_c - acc)[None, :], 0.0)

    M = Minv * beta[None, :]
    #uptull here all prev kernels computions

    gamma=tl.exp(log_gamma)
    k_g=k * gamma#calc gamma
    W=tl.zeros((C, D),dtype=tl.float32)#def U and V
    U=tl.zeros((C,D), dtype=tl.float32)
    for j in range(C):
      m_col_j=tl.sum(tl.where(offs_c[None, :]==j, M,0.0),axis=1)   # (C,)
      kg_j=tl.sum(tl.where(offs_c[:,None]==j, k_g,0.0),axis=0)  # (D,)
      v_j=tl.sum(tl.where(offs_c[:,None]== j, v,0.0), axis=0)  # (D,)
      W =W+ m_col_j[:, None] * kg_j[None, :]
      U =U+ m_col_j[:, None] * v_j[None, :]

    w_ptrs = W_ptr + offs_c[:, None] * D + offs_d[None, :]# U AND W ptrs
    u_ptrs = U_ptr + offs_c[:, None] * D + offs_d[None, :]
    tl.store(w_ptrs, W)#storing caause needed in eqn6,7,8
    tl.store(u_ptrs, U)


def test_build_WU():
    ref = torch.load("step1_reference.pt")
    k, v, alpha, beta = ref["k"], ref["v"], ref["alpha"], ref["beta"]
    W_expected, U_expected = ref["W"], ref["U"]
    C, D = k.shape

    k_cuda=k.cuda().contiguous()
    v_cuda=v.cuda().contiguous()
    alpha_cuda = alpha.cuda().contiguous()
    beta_cuda = beta.cuda().contiguous()
    W_out = torch.zeros(C, D, device="cuda", dtype=torch.float32)
    U_out = torch.zeros(C, D, device="cuda", dtype=torch.float32)

    build_WU_kernel[(1,)](k_cuda, v_cuda, alpha_cuda, beta_cuda, W_out, U_out, C=C, D=D)

    W_out_cpu, U_out_cpu = W_out.cpu(), U_out.cpu()
    w_diff = (W_out_cpu - W_expected).abs().max().item()
    u_diff = (U_out_cpu - U_expected).abs().max().item()

    print("Triton W =\n", W_out_cpu)
    print("Triton U =\n", U_out_cpu)
    print(f"\nW max diff: {w_diff}, close: {torch.allclose(W_out_cpu, W_expected, atol=1e-4)}")
    print(f"U max diff: {u_diff}, close: {torch.allclose(U_out_cpu, U_expected, atol=1e-4)}")


if __name__ == "__main__":
    test_build_WU()
