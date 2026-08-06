import torch

torch.manual_seed(0)
ref = torch.load("step1_reference.pt")
k, v, alpha, beta = ref["k"], ref["v"], ref["alpha"], ref["beta"]
W, U = ref["W"], ref["U"]
C, D = k.shape

log_gamma=torch.log(alpha).cumsum(dim=0)  # (C, D)
gamma=torch.exp(log_gamma)                # (C, D)

S = torch.zeros(D, D, dtype=torch.float32)#init s as 0 for a single chunk

# inter-chunk term: (gamma * q) @ S
q=torch.randn(C, D, dtype=torch.float32)  # need q too, add to saved tensors
torch.manual_seed(0)# reset so q is reproducible

torch.manual_seed(0)
k2=torch.randn(C, D)
v2=torch.randn(C, D)
alpha2=torch.rand(C, D) * 0.9 + 0.1
beta2=torch.rand(C) * 0.5
q2=torch.randn(C, D)# recomp q

# just generate q cleanly alongside everything else
torch.manual_seed(0)
C=8
D=4
k=torch.randn(C,D,dtype=torch.float32)
v=torch.randn(C,D,dtype=torch.float32)
alpha=torch.rand(C,D,dtype=torch.float32)*0.9+0.1
beta=torch.rand(C,dtype=torch.float32)*0.5
q=torch.randn(C, D, dtype=torch.float32)
#same done in step 2 
log_gamma=torch.log(alpha).cumsum(dim=0)
gamma=torch.exp(log_gamma)

# recompute W, U, sometimes inconsistency is there between stages 1 and 2 so recomputing again to be sure
A=torch.zeros(C,C, dtype=torch.float32)
for r in range(C):
    for i in range(r):
        rel=torch.exp(log_gamma[r]-log_gamma[i])
        A[r, i]=beta[r]*(k[r]*rel * k[i]).sum()

I = torch.eye(C, dtype=torch.float32)
M = torch.linalg.inv(I+A) * beta[None, :]
k_g=k * gamma
W=M @ k_g
U=M @ v

#eq 9 with the intra inter and pseudo
S = torch.zeros(D, D, dtype=torch.float32)
pseudo_v =U-W@S#S=0 so pseudo_v = U here

# inter=(gamma * q) @, at beg it will be 0 because S is 0
inter=(q*gamma)@S   # (C, D)

# intra scores: Tril[ (gamma_r/gamma_i) * q_r . k_i ] = Tril[ exp(log_gamma_r - log_gamma_i) * q_r . k_i ]
intra_scores =torch.zeros(C, C,dtype=torch.float32)
for r in range(C):
    for i in range(r + 1):#for diagonal
        rel=torch.exp(log_gamma[r] - log_gamma[i])   
        intra_scores[r, i]=(q[r] * rel * k[i]).sum()

O = inter+intra_scores @ pseudo_v#eqn 9

# S_new = Diag(gamma_C) * S + (Gamma^{i->C} * K)^T @ pseudo_v, eqn 8
gamma_C=gamma[-1]
gamma_rel =torch.exp(log_gamma[-1] -log_gamma)  # (C, D) -- gamma^{i->C} for each i

S_new= S * gamma_C[None, :]   #scales columns
S_new= S_new + (k * gamma_rel).T @ pseudo_v      # (D, D)

torch.set_printoptions(precision=4, sci_mode=False)
print("q =\n", q)
print("\nO =\n", O)
print("\nS_new =\n", S_new)

torch.save({
    "k": k, "v": v, "q": q, "alpha": alpha, "beta": beta,
    "log_gamma": log_gamma, "gamma": gamma,
    "W": W, "U": U, "O": O, "S_new": S_new,
}, "step2_reference.pt")
print("\nSaved to step2_reference.pt")
