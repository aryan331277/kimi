#This is a pytorch reference, and is meant for testing purposes so that triton correctness can be verified
import torch

torch.manual_seed(0)

C = 8# chunk length 
D = 4# key/value dim , have kept it small can change it later,

k=torch.randn(C,D,dtype=torch.float32)
v=torch.randn(C,D,dtype=torch.float32)
alpha=torch.rand(C, D, dtype=torch.float32)*0.9+0.1   # making sure the vals are between 0 and 1
beta=torch.rand(C,dtype=torch.float32)*0.5             # making sure vals b/w 0 and 0.5

log_gamma=torch.log(alpha).cumsum(dim=0)     # defining gamma this will be used in eqn 3/6
gamma=torch.exp(log_gamma)                   # (C, D)

#A: A[r, i] = beta[r] * sum_d exp(log_gamma[r,d]-log_gamma[i,d]) * k[r,d] * k[i,d], for i < r, this is Eq 2/6:Interaction matrix A (strictly lower triangular)
A = torch.zeros(C, C, dtype=torch.float32)
for r in range(C):
    for i in range(r):  
        rel = torch.exp(log_gamma[r] - log_gamma[i])      
        A[r, i] = beta[r] * (k[r] * rel * k[i]).sum()

I = torch.eye(C, dtype=torch.float32)# identity
M = torch.linalg.inv(I + A) * beta[None, :] #taking inverse

k_g=k * gamma#defining W and U from the paper
W=M @ k_g   
U=M @ v      

#print everything so we have real numbers to check Triton against ----
torch.set_printoptions(precision=4, sci_mode=False)
print("k =\n", k)
print("alpha =\n", alpha)
print("beta =\n", beta)
print("log_gamma =\n", log_gamma)
print("A =\n", A)
print("M =\n", M)
print("W =\n", W)
print("U =\n", U)

values
torch.save({
    "k": k, "v": v, "alpha": alpha, "beta": beta,
    "log_gamma": log_gamma, "gamma": gamma,
    "A": A, "M": M, "W": W, "U": U,
}, "step1_reference.pt")
# will be reused later
