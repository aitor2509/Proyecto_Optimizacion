import numpy as np
import matplotlib.pyplot as plt

def _movimiento_exploratorio(func, x, delta):
    xc = x.copy()
    for i in range(len(x)):
        f_actual = func(x)
        x_plus = x.copy(); x_plus[i] += delta[i]
        f_plus = func(x_plus)
        
        if f_plus < f_actual:
            x = x_plus
            continue

        x_minus = x.copy(); x_minus[i] -= delta[i]
        f_minus = func(x_minus)
        if f_minus < f_actual:
            x = x_minus
            
    return x, not np.array_equal(x, xc)

def hooke_jeeves(func, x0, delta, alpha, epsilon):
    x_base = np.array(x0, dtype=float)
    historial = [x_base]
    
    while np.linalg.norm(delta) > epsilon:
        x_nuevo, exito = _movimiento_exploratorio(func, x_base, delta)
        
        if exito:
            # Movimiento de patrón
            x_patron = x_nuevo + (x_nuevo - x_base)
            f_patron = func(x_patron)
            f_nuevo = func(x_nuevo)

            if f_patron < f_nuevo:
                x_base = x_patron
            else:
                x_base = x_nuevo
        else:
            delta /= alpha
        
        historial.append(x_base)
        
    return x_base, historial

def plot_contour(func, history, title):
    hist_array = np.array(history)
    x_min, x_max = hist_array[:, 0].min() - 1, hist_array[:, 0].max() + 1
    y_min, y_max = hist_array[:, 1].min() - 1, hist_array[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    zz = func(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, zz, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Valor de f(x)')
    plt.plot(hist_array[:, 0], hist_array[:, 1], 'r-o', markersize=3, label='Camino')
    plt.title(title); plt.xlabel('x1'); plt.ylabel('x2')
    plt.legend(); plt.grid(True); plt.show()