import numpy as np
import matplotlib.pyplot as plt

def intervalo_mitad(func, a, b, tol=0.001, max_iter=100):
    iterations = []
    a_init, b_init = a, b
    
    for i in range(max_iter):
        c = (a + b) / 2
        d = (c + b) / 2
        e = (a + c) / 2
        
        fc, fd, fe = func(c), func(d), func(e)
        iterations.append((a, e, c, d, b))
        
        if fd < fc: a = c
        elif fe < fc: b = c
        else: a = e; b = d
        
        if abs(b - a) < tol: break
    
    x_min = (a + b) / 2
    f_min = func(x_min)
    return x_min, f_min, iterations, a_init, b_init

def plot_intervalo_mitad(func, x_min, f_min, iterations, a_init, b_init, titulo):
    x = np.linspace(a_init, b_init, 1000)
    y = np.vectorize(func)(x)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', label=f'f(x) = {titulo}')
    ax.plot(x_min, f_min, 'r*', markersize=15, label=f'Mínimo en x={x_min:.4f}')

    for i, (a, e, c, d, b) in enumerate(iterations):
        ax.axvspan(a, b, alpha=0.1, color='yellow', label='Intervalo' if i==0 else "")

    ax.set_title(f"Método de Intervalo Mitad para '{titulo}'")
    ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.grid(True); ax.legend()
    return fig