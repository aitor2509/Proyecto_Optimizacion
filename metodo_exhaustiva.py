import numpy as np
import matplotlib.pyplot as plt

def busqueda_exhaustiva(a, b, n, func):
    delta_x = (b - a) / n
    x1, x2, x3 = a, a + delta_x, a + 2 * delta_x
    iteraciones = []
    
    while x3 <= b:
        f_x1, f_x2, f_x3 = func(x1), func(x2), func(x3)
        iteraciones.append({'x1': x1, 'x2': x2, 'x3': x3, 'f(x1)': f_x1, 'f(x2)': f_x2, 'f(x3)': f_x3})
        
        if f_x1 >= f_x2 <= f_x3:
            return (x1, x3), iteraciones
        
        x1, x2, x3 = x2, x3, x3 + delta_x

    return (a, b) if func(a) < func(b) else (b, a), iteraciones

def graficar_exhaustiva(a, b, func, resultado, titulo):
    x_vals = np.linspace(a, b, 1000)
    y_vals = np.vectorize(func)(x_vals)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, 'b-', label=f'f(x) = {titulo}')
    
    if isinstance(resultado, tuple):
        ax.axvspan(resultado[0], resultado[1], alpha=0.3, color='red', label=f'Intervalo del mínimo: [{resultado[0]:.4f}, {resultado[1]:.4f}]')
    else:
        ax.plot(resultado, func(resultado), 'ro', markersize=8, label=f'Mínimo en x = {resultado:.4f}')
        
    ax.set_title(f"Búsqueda Exhaustiva para '{titulo}'")
    ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.grid(True); ax.legend()
    return fig