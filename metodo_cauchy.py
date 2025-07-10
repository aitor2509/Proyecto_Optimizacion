import numpy as np
import math
import matplotlib.pyplot as plt

def _busqueda_dorada(func, epsilon):
    PHI = (1 + math.sqrt(5)) / 2 - 1
    a, b = 0, 1
    while (b - a) > epsilon:
        x1 = b - PHI * (b - a)
        x2 = a + PHI * (b - a)
        if func(x1) < func(x2): b = x2
        else: a = x1
    return (a + b) / 2

def _gradiente(f, x, delta=1e-5):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += delta
        xn = x.copy(); xn[i] -= delta
        grad[i] = (f(xp) - f(xn)) / (2 * delta)
    return grad

def cauchy(func, x0, epsilon1, epsilon2, max_iter):
    xk = np.array(x0, dtype=float)
    historial = [xk]
    for _ in range(max_iter):
        grad = _gradiente(func, xk)
        if np.linalg.norm(grad) < epsilon1: break
        
        alpha_func = lambda alpha: func(xk - alpha * grad)
        alpha = _busqueda_dorada(alpha_func, epsilon2)
        
        x_k1 = xk - alpha * grad
        if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 1e-8) < epsilon2:
            break
        xk = x_k1
        historial.append(xk)
        
    return xk, historial

def plot_contour(func, history, title):
    pass