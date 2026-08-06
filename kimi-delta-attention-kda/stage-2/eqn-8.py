#eqn 8
@triton.jit
def build_S_new_kernel(
    K_ptr, ALPHA_ptr, U_ptr, W_ptr, S_ptr, S_NEW_ptr,
    C: tl.constexpr, D: tl.constexpr,
):
    offs_c=tl.arange(0, C)
    offs_d=tl.arange(0, D)

    k_ptrs=K_ptr+ offs_c[:,None] * D + offs_d[None,:]
    a_ptrs=ALPHA_ptr+offs_c[:,None]* D + offs_d[None,:]
    u_ptrs=U_ptr+offs_c[:,None] *D + offs_d[ None, :]
    w_ptrs=W_ptr+offs_c[:,None] * D +offs_d[None, :]

    k=tl.load(k_ptrs)
    alpha=tl.load(a_ptrs)
    U=tl.load(u_ptrs)
    W=tl.load(w_ptrs)

    offs_s = tl.arange(0, D)
    s_ptrs=S_ptr +offs_s[:,None]*D+offs_s[None,:]
    S = tl.load(s_ptrs)  

    # log_gamma via manual cumsum
    log_alpha=tl.log(alpha)
    log_gamma=tl.zeros((C, D), dtype = tl.float32)
    acc = tl.zeros((D,), dtype = tl.float32)
    for c in range(C):
        acc = acc + tl.sum(tl.where(offs_c[:, None] == c, log_alpha, 0.0), axis=0)
        log_gamma= log_gamma+tl.where(offs_c[:, None] == c, acc[None, :], 0.0)

    gamma=tl.exp(log_gamma)
    gamma_C=tl.sum(tl.where(offs_c[:, None] == C - 1, log_gamma, 0.0), axis=0)  
    gamma_C=tl.exp(gamma_C)

    # gamma_rel[i] = exp(log_gamma[C-1] - log_gamma[i]) = gamma^{i->C}
    log_gamma_C=tl.sum(tl.where(offs_c[:, None] == C - 1, log_gamma, 0.0),axis=0) 
    gamma_rel=tl.exp(log_gamma_C[None, :] - log_gamma)

    # pseudo_v = U - W @ S  (C, D)
    pseudo_v=tl.zeros((C, D), dtype=tl.float32)
    for j in range(D):
        w_col_j=tl.sum(tl.where(offs_d[None, :] == j, W, 0.0), axis=1)
        s_row_j = tl.sum(tl.where(offs_s[:, None] == j, S, 0.0), axis=0)
        pseudo_v =pseudo_v+ w_col_j[:, None] * s_row_j[None, :]
    pseudo_v = U - pseudo_v

    # S_new = S * gamma_C (scale columns) + (K * gamma_rel)^T @ pseudo_v
    # scale columns: S[d_row, d_col] *= gamma_C[d_col]
    S_new = S * gamma_C[None, :]   

    # += (K * gamma_rel)^T @ pseudo_v  i.e. sum over c: outer(k[c]*gamma_rel[c], pseudo_v[c])
    for c in range(C):
        kg_c=tl.sum(tl.where(offs_c[:, None] == c, k * gamma_rel, 0.0), axis=0)   
        pv_c=tl.sum(tl.where(offs_c[:, None] == c, pseudo_v, 0.0), axis=0)       
        S_new=S_new +kg_c[:, None] * pv_c[None, :]

    sn_ptrs = S_NEW_ptr + offs_s[:, None] * D + offs_s[None, :]
    tl.store(sn_ptrs, S_new)


def test_build_S_new():
    ref = torch.load("step2_reference.pt")
    k, alpha, W, U = ref["k"], ref["alpha"], ref["W"], ref["U"]
    S_new_expected = ref["S_new"]
    C, D = k.shape

    S = torch.zeros(D, D, device="cuda", dtype=torch.float32)

    k_cuda     = k.cuda().contiguous()
    alpha_cuda = alpha.cuda().contiguous()
    U_cuda     = U.cuda().contiguous()
    W_cuda     = W.cuda().contiguous()
    S_new_out  = torch.zeros(D, D, device="cuda", dtype=torch.float32)

    build_S_new_kernel[(1,)](k_cuda, alpha_cuda, U_cuda, W_cuda, S, S_new_out, C=C, D=D)

    S_new_cpu = S_new_out.cpu()
    diff = (S_new_cpu - S_new_expected).abs().max().item()
    print("Triton S_new =\n", S_new_cpu)
    print("\nExpected S_new =\n", S_new_expected)
    print(f"\nMax diff: {diff}")
    print("Close:", torch.allclose(S_new_cpu, S_new_expected, atol=1e-4))


test_build_S_new()
