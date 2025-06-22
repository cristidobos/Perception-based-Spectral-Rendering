from basin_hopping import BasinHoppingOptimizer
from ipopt_optimizer import IPOPT_Optimizer
from optimizer import Optimizer
from evaluate import *


def constrained():
    optimizer = Optimizer('Illuminants/D65.csv')
    res = optimizer.optimize()
    print("Success: ", res.success)
    print("Message: ", res.message)
    print("Value of objective function: ", res.fun)
    print("Iterations:", res.nit)
    optimizer.write_to_file("distributions/D65-2.csv")
    print(res.keys())
    evaluate_optimization(optimizer, "gt-D65-2.png", "color_bars/D65-2.png", "distributions/D65-2.csv")

def ipopt():
    optimizer = IPOPT_Optimizer('Illuminants/Scotopic.csv')
    res = optimizer.optimize()
    print("Success: ", res.success)
    print("Message: ", res.message)
    print("Value of objective function: ", res.fun)
    print("Iterations:", res.nit)
    optimizer.write_to_file("distributions/Scotopic-2.csv")
    evaluate_optimization(optimizer, "Scotopic-gt.png", "color_bars/Scotopic-2.png",
                          "distributions/Scotopic-2.csv")

def basin_hopping():
    optimizer = BasinHoppingOptimizer('spds/resampled_fluorescent.csv')
    res = optimizer.optimize()
    # print("Success: ", res.success)
    print("Message: ", res.message)
    print("Value of objective function: ", res.fun)
    print("Iterations:", res.nit)
    optimizer.write_to_file("distributions/basin-hopping.csv")
    evaluate_optimization(optimizer, "gt-fluorescent-basin.png", "color_bars/color_bar-fluor-basin.png", "distributions/basin-hopping.csv")
    print("\n--- Details of the BEST Local Minimization Run ---")
    best_run_details = res.lowest_optimization_result
    print(f"Was this local run successful? {best_run_details.success}")
    print(f"Message from local run: {best_run_details.message}")
    print(f"Number of iterations in local run: {best_run_details.nit}")


if __name__ == '__main__':
    constrained()

