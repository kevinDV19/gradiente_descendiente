import numpy as np
import matplotlib.pyplot as plt

def mse(Theta0, Theta1, X, y):
    n = len(y)
    error_total = 0

    for i in range(n):
        error_total += (y[i] - (Theta0 * 1 + Theta1 * X[i])) ** 2

    error_promedio = error_total / n
    return error_promedio

def gradiente_descendiente(X, y):
    dimensions = 2
    t = np.array([-5, 5])
    f_range = np.tile(t, (dimensions, 1))

    max_iter = 2000
    num_agents = 1

    agents = np.zeros((num_agents, dimensions))

    for i in range(dimensions):
        dim_f_range = f_range[i, 1] - f_range[i, 0]
        agents[:, i] = np.random.rand(num_agents) * dim_f_range + f_range[i, 0]

    best_position = np.zeros(dimensions)
    best_fitness = np.inf
    fitness = np.empty(num_agents)

    for i in range(num_agents):
        theta0, theta1 = agents[i]
        fitness[i] = mse(theta0, theta1, X, y)
        if fitness[i] < best_fitness:
            best_position = agents[i]
            best_fitness = fitness[i]

    initialPop = agents.copy()
    initialFitness = fitness.copy()

    # Bucle de optimización
    alpha = 0.005  # Tasa de aprendizaje
    delta = 0.001

    for iteration in range(max_iter):
        # Cálculo del gradiente para theta0 y theta1
        gradient_theta0 = (mse(best_position[0] + delta, best_position[1], X, y) - mse(best_position[0], best_position[1], X, y)) / delta
        gradient_theta1 = (mse(best_position[0], best_position[1] + delta, X, y) - mse(best_position[0], best_position[1], X, y)) / delta

        # Actualización de theta0 y theta1
        best_position[0] -= alpha * gradient_theta0
        best_position[1] -= alpha * gradient_theta1

        # Cálculo de la nueva aptitud
        best_fitness = mse(best_position[0], best_position[1], X, y)

    print("Mejor solución: Theta0 =", best_position[0], ", Theta1 =", best_position[1])
    print("Mejor valor de aptitud:", best_fitness)

    xGraph = np.linspace(-5, 5, 25)
    yGraph = np.linspace(-5, 5, 25)
    xv, yv = np.meshgrid(xGraph, yGraph)
    fitnessGraph = np.zeros((25, 25))
    for i in range(25):
        for j in range(25):
            arr = [[xv[i, j], yv[i, j]]]
            fitnessGraph[i, j] = mse(arr[0][0], arr[0][1], X, y)
    plt.ion()
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    plt.title('Función de Regresion Lineal', fontsize=20)
    ax.plot_surface(xv, yv, fitnessGraph, alpha=0.6)
    ax.scatter(initialPop[:, 0], initialPop[:, 1], initialFitness[:], c='green', s=10, marker="x", label='Inicial')
    ax.scatter(agents[:, 0], agents[:, 1], fitness[:], c='red', s=10, marker="o", label='Mejor')
    plt.legend()
    plt.show(block=True)

X = [1, 2, 3, 4] 
y = [1, 2, 3, 4]

gradiente_descendiente(X, y)