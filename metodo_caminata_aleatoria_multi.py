import numpy as np
import matplotlib.pyplot as plt

def caminata_aleatoria_multi(func, x0, max_iter, step_size):
    """
    Algoritmo de Caminata Aleatoria para optimización multivariable.
    
    - func: La función objetivo a minimizar.
    - x0: El punto de inicio (vector numpy).
    - max_iter: Número total de pasos.
    - step_size: El tamaño máximo de cada paso aleatorio.
    """
    x_mejor = np.array(x0, dtype=float)
    f_mejor = func(x_mejor)
    x_actual = x_mejor
    
    history = [x_actual]
    
    for _ in range(max_iter):
        paso = np.random.uniform(-step_size, step_size, size=x_actual.shape)
        x_siguiente = x_actual + paso
        
        if func(x_siguiente) < f_mejor:
            x_mejor = x_siguiente
            f_mejor = func(x_siguiente)
        
        x_actual = x_siguiente
        history.append(x_actual)
    
    return x_mejor, history

def plot_contour(func, history, title):
    hist_array = np.array(history)
    x_min_plot = min(hist_array[:, 0].min(), x_mejor[0]) - 1
    x_max_plot = max(hist_array[:, 0].max(), x_mejor[0]) + 1
    y_min_plot = min(hist_array[:, 1].min(), x_mejor[1]) - 1
    y_max_plot = max(hist_array[:, 1].max(), x_mejor[1]) + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min_plot, x_max_plot, 200), np.linspace(y_min_plot, y_max_plot, 200))
    zz = func(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, zz, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Valor de f(x)')
    
    plt.plot(hist_array[:, 0], hist_array[:, 1], 'w--', alpha=0.3, label='Recorrido')
    plt.plot(hist_array[0, 0], hist_array[0, 1], 'go', markersize=10, label='Inicio')
    plt.plot(x_mejor[0], x_mejor[1], 'y*', markersize=15, label='Mejor punto')
    
    plt.title(title)
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.legend(); plt.grid(True)
    plt.show()

if __name__ == "__main__":
    sphere_func = lambda x: x[0]**2 + x[1]**2
    
    x0 = np.array([10.0, -10.0])
    max_iter = 5000
    step_size = 0.5
    
    x_mejor, historial = caminata_aleatoria_multi(sphere_func, x0, max_iter, step_size)
    
    print(f"Mejor solución encontrada: {x_mejor}")
    print(f"Valor de la función: {sphere_func(x_mejor)}")
    
    plot_contour(lambda p: sphere_func(p.T), historial, "Caminata Aleatoria Multivariable - Función Esfera")