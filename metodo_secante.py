import numpy as np
import matplotlib.pyplot as plt

def secante_min(f, x0, x1, epsilon=0.001, max_iter=100):
    """
    Método de la Secante para optimización.
    Aproxima la raíz de la primera derivada.
    """
    puntos = [x0, x1]
    h = 1e-5 
    f_prime = lambda x: (f(x + h) - f(x - h)) / (2 * h)

    for _ in range(max_iter):
        fpx0 = f_prime(x0)
        fpx1 = f_prime(x1)

        if abs(fpx1 - fpx0) < 1e-10: # Evitar división por cero
            break
            
        x2 = x1 - fpx1 * (x1 - x0) / (fpx1 - fpx0)
        puntos.append(x2)

        if abs(x2 - x1) < epsilon:
            break
        x0, x1 = x1, x2
        
    return x2, puntos

def graficar_secante(f, x_range, puntos, xmin, titulo):
    x_vals = np.linspace(x_range[0], x_range[1], 500)
    y_vals = np.vectorize(f)(x_vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, 'b-', label=f'f(x) = {titulo}')
    ax.plot(xmin, f(xmin), 'r*', markersize=15, label=f'Mínimo en x={xmin:.4f}')

    y_puntos = [f(p) for p in puntos]
    ax.plot(puntos, y_puntos, 'go--', alpha=0.7, label='Pasos de Secante')
    ax.scatter(puntos, y_puntos, c='green', alpha=0.7)
    
    ax.set_title(f"Método de la Secante para '{titulo}'")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.grid(True); ax.legend()
    return fig