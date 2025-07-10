import numpy as np
import matplotlib.pyplot as plt

def nelder_mead(func, x0, alpha=1, gamma=2, rho=0.5, sigma=0.5, epsilon=1e-5, max_iter=1000):
    dim = len(x0)
    simplex = [np.array(x0, dtype=float)]
    historial = [np.array(x0, dtype=float)]
    
    for i in range(dim):
        point = np.array(x0, dtype=float)
        point[i] += alpha
        simplex.append(point)
    
    for i in range(max_iter):
        scores = [(s, func(s)) for s in simplex]
        scores.sort(key=lambda x: x[1])
        
        best, worst = scores[0][0], scores[-1][0]
        historial.append(best)
        
        if np.std([s[1] for s in scores]) < epsilon:
            break
            
        centroid = np.mean([s[0] for s in scores[:-1]], axis=0)
        
        xr = centroid + alpha * (centroid - worst)
        if scores[0][1] <= func(xr) < scores[-2][1]:
            simplex[-1] = xr
            continue
            
        if func(xr) < scores[0][1]:
            xe = centroid + gamma * (xr - centroid)
            if func(xe) < func(xr):
                simplex[-1] = xe
            else:
                simplex[-1] = xr
            continue

        xc = centroid + rho * (worst - centroid)
        if func(xc) < func(worst):
            simplex[-1] = xc
            continue

        for j in range(1, len(simplex)):
            simplex[j] = scores[0][0] + sigma * (simplex[j] - scores[0][0])
            
    return historial[-1], historial

def plot_contour(func, history, title):
    pass