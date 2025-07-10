import numpy as np
import matplotlib.pyplot as plt

def hill_climbing_multi(func, x0, max_iter, step_size):
    """
    Algoritmo de Hill Climbing para optimización multivariable.
    
    - func: La función objetivo a minimizar.
    - x0: El punto de inicio (vector numpy).
    - max_iter: Número máximo de iteraciones.
    - step_size: El tamaño del paso para generar vecinos.
    """
    x_current = np.array(x0, dtype=float)
    f_current = func(x_current)
    history = [x_current]

    for _ in range(max_iter):
        x_next = x_current + np.random.uniform(-step_size, step_size, size=x_current.shape)
        f_next = func(x_next)
        
        if f_next < f_current:
            x_current = x_next
            f_current = f_next
        
        history.append(x_current)
    
    return x_current, history

def plot_contour(func, history, title):
    """Grafica el historial de optimización en un mapa de contornos."""
    hist_array = np.array(history)
    x_min, x_max = hist_array[:, 0].min() - 1, hist_array[:, 0].max() + 1
    y_min, y_max = hist_array[:, 1].min() - 1, hist_array[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    zz = func(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, zz, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Valor de f(x)')
    
    plt.plot(hist_array[:, 0], hist_array[:, 1], 'r-o', markersize=3, label='Camino')
    plt.plot(hist_array[0, 0], hist_array[0, 1], 'go', markersize=10, label='Inicio')
    plt.plot(hist_array[-1, 0], hist_array[-1, 1], 'y*', markersize=15, label='Fin')
    
    plt.title(title)
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.legend(); plt.grid(True)
    plt.show()

if __name__ == "__main__":
    booth_func = lambda x: (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    
    x0 = np.array([-5.0, 8.0])
    max_iter = 2000
    step_size = 0.1
    
    solucion, historial = hill_climbing_multi(booth_func, x0, max_iter, step_size)
    
    print(f"Solución encontrada: {solucion}")
    print(f"Valor de la función: {booth_func(solucion)}")
    
    plot_contour(lambda p: booth_func(p.T), historial, "Hill Climbing Multivariable - Función de Booth")