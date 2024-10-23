import cvxpy as cp

if __name__ == '__main__':
    nx = 4
    hat_Sigma_minus, hat_Sigma_plus = cp.Variable((nx,nx), symmetric=True), cp.Variable((nx,nx), symmetric=True)
    W1, W2 = cp.Variable((nx,nx), symmetric=True), cp.Variable((nx,nx), symmetric=True)
    Sig0, SigT = cp.Parameter((nx, nx), PSD=True), cp.Parameter((nx, nx), PSD=True) 
    Phi1, Phi2 = cp.Parameter((nx, nx), PSD=True), cp.Parameter((nx, nx), PSD=True)
    S1, S2 = cp.Parameter((nx, nx), PSD=True), cp.Parameter((nx, nx), PSD=True)
    logdet_1 = -cp.log_det(hat_Sigma_minus - W1@Sig0@cp.transpose(W1))
    trace_1 = cp.trace(hat_Sigma_minus)
    
    print("curvature of logdet_1:", logdet_1.curvature)
    print("curvature of trace_1:", trace_1.curvature)
