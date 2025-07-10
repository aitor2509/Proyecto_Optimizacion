import numpy as np
import matplotlib.pyplot as plt

def golden_section_search(func, a, b, tol=0.001, max_iter=100):
    tau = (np.sqrt(5) - 1) / 2
    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)
    f1 = func(x1)
    f2 = func(x2)
    iterations = [(a, x1, x2, b, f1, f2)]
    
    for i in range(max_iter):
        if f1 > f2:
            a = x1; x1 = x2; f1 = f2; x2 = a + tau * (b - a); f2 = func(x2)
        else:
            b = x2; x2 = x1; f2 = f1; x1 = b - tau * (b - a); f1 = func(x1)
        iterations.append((a, x1, x2, b, f1, f2))
        if abs(b - a) < tol: break
    
    x_min = (a + b) / 2
    f_min = func(x_min)
    return x_min, f_min, iterations

def plot_golden(func, x_min, f_min, iterations, a_init, b_init, titulo):
    x = np.linspace(a_init, b_init, 1000)
    y = np.vectorize(func)(x)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', label=f'f(x) = {titulo}')
    ax.plot(x_min, f_min, 'r*', markersize=15, label=f'Mínimo en x={x_min:.4f}')
    
    for i, (a, x1, x2, b, f1, f2) in enumerate(iterations):
        ax.plot(x1, f1, 'go', markersize=5, alpha=0.6, label='x1' if i==0 else "")
        ax.plot(x2, f2, 'yo', markersize=5, alpha=0.6, label='x2' if i==0 else "")
        
    ax.set_title(f"Método de Sección Dorada para '{titulo}'")
    ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.grid(True); ax.legend()
    return fig