import numpy as np
import matplotlib.pyplot as plt

def recocido_simulado_multi(func, x0, temp_inicial, alpha, max_iter):
    """
    Algoritmo de Recocido Simulado para optimización multivariable.
    
    - func: La función objetivo.
    - x0: Punto de inicio (vector).
    - temp_inicial: Temperatura inicial.
    - alpha: Factor de enfriamiento (ej. 0.95).
    - max_iter: Número de iteraciones por cada nivel de temperatura.
    """
    x_actual = np.array(x0, dtype=float)
    mejor_solucion = x_actual
    f_actual = func(x_actual)
    f_mejor = f_actual
    temp = temp_inicial
    
    history = [x_actual]
    
    for _ in range(max_iter):
        vecino = x_actual + np.random.randn(len(x_actual)) * 0.5
        f_vecino = func(vecino)
        
        delta = f_vecino - f_actual
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            x_actual = vecino
            f_actual = f_vecino
        
        if f_actual < f_mejor:
            mejor_solucion = x_actual
            f_mejor = f_actual
        
        history.append(mejor_solucion)
        
        temp *= alpha
        
    return mejor_solucion, history

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
    plt.plot(hist_array[0, 0], hist_array[0, 1], 'go', markersize=10, label='Inicio')
    plt.plot(hist_array[-1, 0], hist_array[-1, 1], 'y*', markersize=15, label='Fin')
    
    plt.title(title)
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.legend(); plt.grid(True)
    plt.show()

if __name__ == "__main__":
    himmelblau_func = lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    
    x0 = np.array([0.0, 0.0])
    temp_inicial = 100.0
    alpha = 0.99
    max_iter = 5000
    
    solucion, historial = recocido_simulado_multi(himmelblau_func, x0, temp_inicial, alpha, max_iter)
    
    print(f"Solución encontrada: {solucion}")
    print(f"Valor de la función: {himmelblau_func(solucion)}")
    
    plot_contour(lambda p: himmelblau_func(p.T), historial, "Recocido Simulado - Función de Himmelblau")