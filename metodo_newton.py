import numpy as np
import matplotlib.pyplot as plt

def newton_raphson_optimization(df, ddf, x0, epsilon=0.001, max_iter=100):
    """
    Método de Newton-Raphson para optimización.
    Encuentra el mínimo de una función buscando la raíz de su primera derivada.
    Usa la fórmula: x_new = x - f'(x) / f''(x)
    """
    x = x0
    pasos = [x]
    for i in range(max_iter):
        dfx = df(x)
        ddfx = ddf(x)
        
        if abs(ddfx) < 1e-8: # Evitar división por cero si la segunda derivada es plana
            print("Segunda derivada muy cercana a cero. Método detenido.")
            break
            
        x_new = x - dfx / ddfx
        pasos.append(x_new)
        
        if abs(x_new - x) < epsilon:
            break
        x = x_new
        
    return x, pasos

def plot_newton(func, xmin, pasos, titulo):
    x_min_range = min(pasos) - 2
    x_max_range = max(pasos) + 2
    x_vals = np.linspace(x_min_range, x_max_range, 1000)
    y_vals = np.vectorize(func)(x_vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, 'b-', label=f'f(x) = {titulo}')
    ax.plot(xmin, func(xmin), 'r*', markersize=15, label=f'Mínimo en x={xmin:.4f}')
    
    y_pasos = [func(p) for p in pasos]
    ax.plot(pasos, y_pasos, 'go--', alpha=0.7, label='Pasos de Newton')
    ax.scatter(pasos, y_pasos, c='green', alpha=0.7)

    ax.set_title(f"Método de Newton-Raphson para '{titulo}'")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.grid(True); ax.legend()
    return fig