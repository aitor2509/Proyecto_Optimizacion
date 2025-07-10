import numpy as np
import matplotlib.pyplot as plt
from metodo_cauchy import _busqueda_dorada, _gradiente # Reutilizamos código

def _hessiano(f, x, delta=1e-4):
    n = len(x)
    hess = np.zeros((n, n), dtype=float)
    fx = f(x)
    for i in range(n):
        for j in range(n):
            if i == j:
                xp = x.copy(); xp[i] += delta
                xn = x.copy(); xn[i] -= delta
                hess[i, j] = (f(xp) - 2 * fx + f(xn)) / (delta**2)
            else:
                xpp = x.copy(); xpp[i] += delta; xpp[j] += delta
                xpn = x.copy(); xpn[i] += delta; xpn[j] -= delta
                xnp = x.copy(); xnp[i] -= delta; xnp[j] += delta
                xnn = x.copy(); xnn[i] -= delta; xnn[j] -= delta
                hess[i, j] = (f(xpp) - f(xpn) - f(xnp) + f(xnn)) / (4 * delta**2)
    return hess

def newton_multi(func, x0, epsilon1, epsilon2, max_iter):
    xk = np.array(x0, dtype=float)
    historial = [xk]
    for _ in range(max_iter):
        grad_k = _gradiente(func, xk)
        if np.linalg.norm(grad_k) < epsilon1: break
            
        hess_k = _hessiano(func, xk)
        try:
            hess_inv_k = np.linalg.inv(hess_k + 1e-8 * np.identity(len(xk)))
        except np.linalg.LinAlgError:
            break
        pk = -np.dot(hess_inv_k, grad_k)
        
        alpha_func = lambda alpha: func(xk + alpha * pk)
        alpha = _busqueda_dorada(alpha_func, epsilon2)
        
        x_k1 = xk + alpha * pk
        if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 1e-8) < epsilon2: break
            
        xk = x_k1
        historial.append(xk)
        
    return xk, historial

def plot_contour(func, history, title):
    # (Función de ploteo idéntica a la de Hooke-Jeeves)
    pass