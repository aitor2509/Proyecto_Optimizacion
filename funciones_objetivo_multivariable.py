import numpy as np


FUNCIONES_MULTI = {
    "Rastrigin": {
        "func": lambda x: 10 * 2 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])),
        "grad": lambda x: np.array([2*x[0] + 20*np.pi*np.sin(2*np.pi*x[0]), 2*x[1] + 20*np.pi*np.sin(2*np.pi*x[1])]),
        "latex": r"f(x, y) = 20 + (x^2 - 10 \cos(2\pi x)) + (y^2 - 10 \cos(2\pi y))",
        "dominio": "[-5.12, 5.12] x [-5.12, 5.12]",
        "minimo": "f(0, 0) = 0",
        "x0": np.array([1.0, 1.0])
    },
    "Ackley": {
        "func": lambda x: -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20,
        "grad": lambda x: np.array([
            (2 * x[0] * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))) / (np.sqrt(0.5 * (x[0]**2 + x[1]**2)) + 1e-9) + np.pi * np.sin(2 * np.pi * x[0]) * np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))),
            (2 * x[1] * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))) / (np.sqrt(0.5 * (x[0]**2 + x[1]**2)) + 1e-9) + np.pi * np.sin(2 * np.pi * x[1]) * np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
        ]),
        "latex": r"f(x,y) = -20e^{-0.2\sqrt{0.5(x^2+y^2)}} - e^{0.5(\cos(2\pi x)+\cos(2\pi y))} + e + 20",
        "dominio": "[-5, 5] x [-5, 5]",
        "minimo": "f(0, 0) = 0",
        "x0": np.array([1.0, 1.0])
    },
    "Sphere": {
        "func": lambda x: x[0]**2 + x[1]**2,
        "grad": lambda x: np.array([2*x[0], 2*x[1]]),
        "latex": r"f(x, y) = x^2 + y^2",
        "dominio": "[-10, 10] x [-10, 10]",
        "minimo": "f(0, 0) = 0",
        "x0": np.array([10.0, -10.0])
    },
    "Rosenbrock": {
        "func": lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2,
        "grad": lambda x: np.array([-400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]), 200*(x[1]-x[0]**2)]),
        "latex": r"f(x, y) = 100(y - x^2)^2 + (1 - x)^2",
        "dominio": "[-2, 2] x [-1, 3]",
        "minimo": "f(1, 1) = 0",
        "x0": np.array([0.0, 0.0])
    },
    "Beale": {
        "func": lambda x: (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2,
        "grad": lambda x: np.array([
            2*(1.5 - x[0] + x[0]*x[1])*(-1 + x[1]) + 2*(2.25 - x[0] + x[0]*x[1]**2)*(-1 + x[1]**2) + 2*(2.625 - x[0] + x[0]*x[1]**3)*(-1 + x[1]**3),
            2*(1.5 - x[0] + x[0]*x[1])*(x[0]) + 2*(2.25 - x[0] + x[0]*x[1]**2)*(2*x[0]*x[1]) + 2*(2.625 - x[0] + x[0]*x[1]**3)*(3*x[0]*x[1]**2)
        ]),
        "latex": r"f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2",
        "dominio": "[-4.5, 4.5] x [-4.5, 4.5]",
        "minimo": "f(3, 0.5) = 0",
        "x0": np.array([1.0, 1.0])
    },
    "Booth": {
        "func": lambda x: (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2,
        "grad": lambda x: np.array([2*(x[0] + 2*x[1] - 7) + 4*(2*x[0] + x[1] - 5), 4*(x[0] + 2*x[1] - 7) + 2*(2*x[0] + x[1] - 5)]),
        "latex": r"f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2",
        "dominio": "[-10, 10] x [-10, 10]",
        "minimo": "f(1, 3) = 0",
        "x0": np.array([-5.0, 8.0])
    },
    "Himmelblau": {
        "func": lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2,
        "grad": lambda x: np.array([4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7), 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)]),
        "latex": r"f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2",
        "dominio": "[-5, 5] x [-5, 5]",
        "minimo": "Varios mínimos, ej: f(3, 2) = 0",
        "x0": np.array([0.0, 0.0])
    },
    "McCormick": {
        "func": lambda x: np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1,
        "grad": lambda x: np.array([np.cos(x[0] + x[1]) + 2*(x[0] - x[1]) - 1.5, np.cos(x[0] + x[1]) - 2*(x[0] - x[1]) + 2.5]),
        "latex": r"f(x, y) = \sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1",
        "dominio": "[-1.5, 4] x [-3, 4]",
        "minimo": "f(-0.547, -1.547) ≈ -1.913",
        "x0": np.array([0.0, 0.0])
    }
}