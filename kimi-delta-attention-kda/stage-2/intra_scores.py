#The code here is more or less the same as stage 1 part 2 the major difference in this part of the code stage 1 part 2 is that here the score gets computed and intra_scores are calculated in s1 p2 A matrix was built so apart from beta changes and low tri scaling it is mostly the same
import torch
import triton
import triton.language as tl

#check the row calculation and see the difference in what is being multiplied to get the row answer

@triton.jit
def build_intra_scores_kernel(
    Q_ptr, K_ptr, ALPHA_ptr,
    SCORES_ptr,
    C: tl.constexpr, D: tl.constexpr,
):
    offs_c = tl.arange(0, C)
    offs_d = tl.arange(0, D)

    q_ptrs = Q_ptr + offs_c[:, None] * D + offs_d[None, :]
    k_ptrs = K_ptr + offs_c[:, None] * D + offs_d[None, :]
    a_ptrs = ALPHA_ptr + offs_c[:, None] * D + offs_d[None, :]

    q = tl.load(q_ptrs)     
    k = tl.load(k_ptrs)      
    alpha = tl.load(a_ptrs)  

    log_alpha = tl.log(alpha)
    log_gamma = tl.zeros((C, D), dtype=tl.float32)
    acc = tl.zeros((D,), dtype=tl.float32)
    for c in range(C):
        acc = acc + tl.sum(tl.where(offs_c[:, None] == c, log_alpha, 0.0), axis=0)
        log_gamma += tl.where(offs_c[:, None] == c, acc[None, :], 0.0)

    scores = tl.zeros((C, C), dtype=tl.float32)
    for r in range(C):
        lg_r = tl.sum(tl.where(offs_c[:, None] == r, log_gamma, 0.0), axis=0)  
        q_r  = tl.sum(tl.where(offs_c[:, None] == r, q, 0.0), axis=0)          
        rel  = tl.exp(lg_r[None, :] - log_gamma)                               
        row  = tl.sum(q_r[None, :] * rel * k, axis=1)                          
        row  = tl.where(offs_c <= r, row, 0.0)                                  
        scores += tl.where(offs_c[:, None] == r, row[None, :], 0.0)

    s_ptrs = SCORES_ptr + offs_c[:, None] * C + offs_c[None, :]
    tl.store(s_ptrs, scores)


def test_intra_scores():
    ref = torch.load("step2_reference.pt")
    q, k, alpha = ref["q"], ref["k"], ref["alpha"]
    log_gamma = ref["log_gamma"]
    C, D = k.shape

    expected = torch.zeros(C, C)
    for r in range(C):
        for i in range(r + 1):
            rel = torch.exp(log_gamma[r] - log_gamma[i])
            expected[r, i] = (q[r] * rel * k[i]).sum()

    q_cuda     = q.cuda().contiguous()
    k_cuda     = k.cuda().contiguous()
    alpha_cuda = alpha.cuda().contiguous()
    scores_out = torch.zeros(C, C, device="cuda", dtype=torch.float32)

    build_intra_scores_kernel[(1,)](q_cuda, k_cuda, alpha_cuda, scores_out, C=C, D=D)

    scores_cpu = scores_out.cpu()
    diff = (scores_cpu - expected).abs().max().item()
    print("Triton intra scores =\n", scores_cpu)
    print("\nExpected =\n", expected)
    print(f"\nMax diff: {diff}")
    print("Close:", torch.allclose(scores_cpu, expected, atol=1e-4))


test_intra_scores()
