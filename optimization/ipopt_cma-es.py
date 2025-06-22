import time
import secrets
import jax

import jax.numpy as jnp
from jax.nn import softmax
from jax import jit, vmap
from evosax.algorithms import CMA_ES
from color import *
from util import *
from evaluate import *

from ipopt_optimizer import IPOPT_Optimizer

# export LD_PRELOAD=/usr/local/cuda-12.9/lib64/libcublas.so.12:\
# /usr/local/cuda-12.9/lib64/libcusolver.so.11
#
#  export LD_LIBRARY_PATH=/home/cristi/dev/hsl/lib64/libhsl.so:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# export XLA_FLAGS="--xla_gpu_graph_level=0"   # avoid graph-capture bug on 56x drivers
# export JAX_ENABLE_X64=1                      # keep CMA-ES numerically stable
#
# python ipopt_cma-es.py

ipopt_optimizer = IPOPT_Optimizer('spds/factored_fluorescent.csv')
population_size = 230
num_generations = 5000

N = ipopt_optimizer.N
lambdas = ipopt_optimizer.lambdas
TOTAL_VARS = N * len(lambdas)
lambda_min = ipopt_optimizer.lambda_min
lambda_max = ipopt_optimizer.lambda_max
lambda_step = ipopt_optimizer.lambda_step
U = ipopt_optimizer.U
I = ipopt_optimizer.I
Lab_gt_ref = ipopt_optimizer.Lab_gt_ref
x_bar = ipopt_optimizer.x_bar
y_bar = ipopt_optimizer.y_bar
z_bar = ipopt_optimizer.z_bar
initial = ipopt_optimizer.initial_densities

def core(flat):
    raw = flat.reshape((N, len(lambdas)))
    pdfs = softmax(raw, axis=1)

    cdfs = jnp.stack([
        cumulative_trapezoid_jax(pdfs[i], lambdas, initial=0.0)
        for i in range(N)
    ])

    xs = jnp.linspace(0, 1.0, U)

    lmb = jnp.vstack([
        jnp.interp(xs, cdfs[i], lambdas)
        for i in range(N)
    ])

    h_x = jnp.vstack([
        jnp.interp(lmb[i], lambdas, x_bar)
        for i in range(N)
    ])
    h_y = jnp.vstack([
        jnp.interp(lmb[i], lambdas, y_bar)
        for i in range(N)
    ])
    h_z = jnp.vstack([
        jnp.interp(lmb[i], lambdas, z_bar)
        for i in range(N)
    ])

    illum = jnp.vstack([
        jnp.interp(lmb[i], lambdas, I)
        for i in range(N)
    ])

    denom = jnp.stack([
        jnp.sum(
            jnp.vstack([
                jnp.interp(lmb[i], lambdas, pdfs[k])
                for k in range(N)
            ]), axis=0
        )
        for i in range(N)
    ])
    denom = jnp.maximum(denom, 1e-8)

    sum_x = jnp.sum(h_x * illum / denom, axis=0)
    sum_y = jnp.sum(h_y * illum / denom, axis=0)
    sum_z = jnp.sum(h_z * illum / denom, axis=0)

    labs = vmap(lambda X, Y, Z: XYZ_to_Lab_jax(jnp.array([X, Y, Z])))(sum_x, sum_y, sum_z)

    dEs = vmap(lambda lab: delta_E_CIE2000_jax(lab, Lab_gt_ref))(labs)

    final_error = jnp.trapezoid(dEs, xs)

    return final_error

objective = jit(core)

eps = 1e-12
initial_logits = jnp.log(initial + eps)       # inverse softmax
x0             = initial_logits.ravel()           # flat vector for CMA-ES

x0 = initial.ravel()

strategy = CMA_ES(population_size=population_size, solution=x0)
params = strategy.default_params.replace(std_init = 2.5,
                                            std_min = 1e-6,
                                            c_std = 0.18,
                                            d_std = 2.5)

seed = secrets.randbits(32)
key = jax.random.PRNGKey(seed)
state = strategy.init(key, x0, params)

def step(key, state):
    key, ask_key, tell_key = jax.random.split(key, 3)
    y_population, state = strategy.ask(ask_key, state, params)

    fitness_scores = jax.vmap(objective)(y_population)

    state, metrics = strategy.tell(tell_key, y_population, fitness_scores, state, params)
    return key, state, metrics

generation = jit(step)

print("Starting CMA-ES optimization...")
start_time = time.time()

best         = jnp.inf
plateau_gens = 0
PLATEAU_MAX  = 100        # generations without significant progress
TOL          = 1e-3

for i in range(num_generations):
    key, state, metrics = generation(key, state)
    if metrics["best_fitness"] < best - TOL:
        best, plateau_gens = metrics["best_fitness"], 0
    else:
        plateau_gens += 1

    if plateau_gens >= PLATEAU_MAX:
        print(f"[restart] gen {i:4d}  σ→{state.std:.2e}")
        # inflate σ 5× and re-seed at current best
        new_std = 2.5 * (num_generations - i) / num_generations
        params = params.replace(std_init=state.std * 2.0)
        state = strategy.init(key, state.best_solution, params)
        plateau_gens = 0

    if i % 10 == 0:
        print(f"generation = {i:5d} | best = {metrics['best_fitness']:.5f} | σ = {state.std:.3e}")

end_time = time.time()
print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")

best_solution = state.best_solution
final_distributions = jax.nn.softmax(best_solution.reshape(N, len(lambdas)), axis=1)

print("Starting second optimization step using IPOPT...")
ipopt_optimizer.set_initial_densities(final_distributions)
result = ipopt_optimizer.optimize()

print("Success: ", result.success)
print("Message: ", result.message)
print("Value of objective function: ", result.fun)
print("Iterations:", result.nit)

ipopt_optimizer.write_to_file("ipopt_cma-es.csv")
evaluate_optimization(ipopt_optimizer, "", "color_bars/color-bar-ipopt-cma.png",
                          "distributions/pdf_ipopt_cma.csv")

