#eqn 9
@triton.jit
def build_O_kernel(
    Q_ptr, K_ptr, ALPHA_ptr, U_ptr, W_ptr, S_ptr, O_ptr,
    C: tl.constexpr, D: tl.constexpr,
):
    offs_c=tl.arange(0, C)
    offs_d=tl.arange(0, D)

    q_ptrs=Q_ptr+offs_c[:,None]*D+offs_d[None,:]
    k_ptrs=K_ptr+offs_c[:,None]*D+offs_d[None,:]
    a_ptrs=ALPHA_ptr+offs_c[:,None]*D+offs_d[None,:]
    u_ptrs=U_ptr+offs_c[:,None]*D+offs_d[None,:]
    w_ptrs=W_ptr+offs_c[:,None]*D+offs_d[None,:]

    q= tl.load(q_ptrs)
    k=tl.load(k_ptrs)
    alpha=tl.load(a_ptrs)
    U= tl.load(u_ptrs)   
    W= tl.load(w_ptrs)   

    offs_s= tl.arange(0, D)
    s_ptrs= S_ptr + offs_s[:, None] * D + offs_s[None, :]
    S= tl.load(s_ptrs)       

    # log_gamma via manual cumsum
    log_alpha=tl.log(alpha)
    log_gamma = tl.zeros((C, D),dtype=tl.float32)
    acc = tl.zeros((D,), dtype=tl.float32)
    for c in range(C):
        acc = acc + tl.sum(tl.where(offs_c[:, None] == c, log_alpha, 0.0), axis=0)
        log_gamma=log_gamma+ tl.where(offs_c[:, None] == c, acc[None, :], 0.0)

    gamma = tl.exp(log_gamma)  # (C, D)

#ALL THIS IS ALREADY DONE IN THE PREVIOUS VERSIONS DONE HERE AGAIN TO DEBUG AND UNDERSTAND EASILY
    # pseudo_v = U - W @ S  (C, D)
    pseudo_v=tl.zeros((C, D),dtype=tl.float32)
    for j in range(D):
        w_col_j=tl.sum(tl.where(offs_d[None,:]==j, W,0.0),axis=1)   # (C,)
        s_row_j= tl.sum(tl.where(offs_s[:,None]==j,S,0.0),axis=0)   # (D,)
        pseudo_v=pseudo_v+ w_col_j[:, None]*s_row_j[None, :]
    pseudo_v=U- pseudo_v

    # inter: (gamma * q) @ S  (C, D)
    gamma_q = q * gamma
    inter = tl.zeros((C, D), dtype=tl.float32)
    for j in range(D):
        gq_col_j=tl.sum(tl.where(offs_d[None,:]==j, gamma_q,0.0),axis=1)  # (C,)
        s_row_j =tl.sum(tl.where(offs_s[:,None]==j,S,0.0),axis=0)         # (D,)
        inter=inter+ gq_col_j[:, None] * s_row_j[None, :]

    # intra scores (C, C) lower triangular, again previously computed in stage 2 part 2  
    scores = tl.zeros((C, C), dtype=tl.float32)
    for r in range(C):
        lg_r = tl.sum(tl.where(offs_c[:, None] == r, log_gamma, 0.0), axis=0)
        q_r  = tl.sum(tl.where(offs_c[:, None] == r, q, 0.0), axis=0)
        rel  = tl.exp(lg_r[None, :] - log_gamma)
        row  = tl.sum(q_r[None, :] * rel * k, axis=1)
        row  = tl.where(offs_c <= r, row, 0.0)
        scores += tl.where(offs_c[:, None] == r, row[None, :], 0.0)

    # intra = scores @ pseudo_v  (C, D)
    intra = tl.zeros((C, D), dtype=tl.float32)
    for j in range(C):
        sc_col_j  = tl.sum(tl.where(offs_c[None, :] == j, scores, 0.0), axis=1)  # (C,)
        pv_row_j  = tl.sum(tl.where(offs_c[:, None] == j, pseudo_v, 0.0), axis=0) # (D,)
        intra += sc_col_j[:, None] * pv_row_j[None, :]

    O = inter + intra

    o_ptrs = O_ptr + offs_c[:, None] * D + offs_d[None, :]
    tl.store(o_ptrs, O)


def test_build_O():
    ref = torch.load("step2_reference.pt")
    q, k, alpha = ref["q"], ref["k"], ref["alpha"]
    W, U, O_expected = ref["W"], ref["U"], ref["O"]
    C, D = k.shape

    S = torch.zeros(D, D, device="cuda", dtype=torch.float32)

    q_cuda     = q.cuda().contiguous()
    k_cuda     = k.cuda().contiguous()
    alpha_cuda = alpha.cuda().contiguous()
    U_cuda     = U.cuda().contiguous()
    W_cuda     = W.cuda().contiguous()
    O_out      = torch.zeros(C, D, device="cuda", dtype=torch.float32)

    build_O_kernel[(1,)](q_cuda, k_cuda, alpha_cuda, U_cuda, W_cuda, S, O_out, C=C, D=D)

    O_cpu = O_out.cpu()
    diff  = (O_cpu - O_expected).abs().max().item()
    print("Triton O =\n", O_cpu)
    print("\nExpected O =\n", O_expected)
    print(f"\nMax diff: {diff}")
    print("Close:", torch.allclose(O_cpu, O_expected, atol=1e-4))


test_build_O()
