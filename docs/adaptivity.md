---
title: Adaptive control of simulations
permalink: tooling-micro-manager-adaptivity.html
keywords: tooling, macro-micro, two-scale, adaptivity
summary: Micro Manager can solve simulations adaptively.
---

## Main concept

In most two-scale scenarios, fully resolving the finer (micro) scale is oftentimes not feasible due to computational restrictions. Several adaptivity techniques to reduce the required micro scale problems to compute exist. The adaptivity strategy implemented in the Micro Manager is originally proposed in

Redeker, Magnus & Eck, Christof. (2013). A fast and accurate adaptive solution strategy for two-scale models with continuous inter-scale dependencies. Journal of Computational Physics. 240. 268-283. [10.1016/j.jcp.2012.12.025](https://doi.org/10.1016/j.jcp.2012.12.025).

and further improved in

Bastidas, Manuela & Bringedal, Carina & Pop, Iuliu Sorin. (2021). A two-scale iterative scheme for a phase-field model for precipitation and dissolution in porous media. Applied Mathematics and Computation. 396. 125933. [10.1016/j.amc.2020.125933](https://doi.org/10.1016/j.amc.2020.125933).

In the time step $t_n$, the adaptivity computation consists of the following steps

1. Update the similarity distance matrix

    For a quantity $u$, a similarity distance $d_{u}$ is calculated in the following way

    $d_{u}(x_{1}, x_{2}; t_{n}) := ||u(x_{1}, t_{n}) - u(x_{2}, t_{n})||$

    If $u$ is a vector, the distance is calculated component-wise by

    $d_{u}(x_{1}, x_{2}; t_{n}) := \sum_{i=1}^{d} ||u_{i}(x_{1}, t_{n}) - u_{i}(x_{2}, t_{n})||$

    Calculate entries of the similarity distance matrix $D$ between each micro simulation pair

    $D(x_{1}, x_{2}; t_{n}; \Lambda) \approx e^{-\Lambda \Delta t} D(x_{1}, x_{2}; t_{n-1}; \Lambda) + \Delta t (d_{u}(x_{1}, x_{2}; t_{n}))$

2. Update the set of active simulations $N_{A}(t_{n})$.

    For each active simulation $x_{A} \in N_{A}(t_{n-1})$, we check if there exists another active simulation $x_{B} \in N_{A}(t_{n-1})$ such that

    $D(x_{A}, x_{B}; t_{n-1}; \Lambda) \le tol_{c}$

    and if it is the case, $x_{A}$ is deactivated.

3. Update the set of inactive simulations $N_{I}(t_{n})$. For each inactive simulations $x_{I} \in N_{I}(t_{n-1})$, gather the distance $D$ to all active simulations. If

    $\displaystyle\min_{x_{A} \in N_{A}(n)} {D(x_{I}, x_{A}; t_{n-1}; \Lambda)} > tol_{r}$

    the simulation $x_{I}$ is activated.

4. Associate each inactive simulation to its most similar active simulation. An inactive simulation $x_{I} \in N_{I}(t_{n})$ is associated with an active simulation $x_{A} \in N_{A}(t_{n})$ if

    $x_{A} = \displaystyle\min_{x \in N_{A}(t_{n})} {D(x_{I}, x; t_{n-1}; \Lambda)}$

These steps are repeated every time the adaptivity computation is triggered.

The coarsening tolerance $t_c$ is calculated by

$t_c = C_c C_r \displaystyle\max_{x_{1},x_{2} \in N(t_{n})} {D(x_{1},x_{2}; t_{n})}$

The refining tolerance $t_r$ is calculated by

$t_r = C_r \displaystyle\max_{x_{1},x_{2} \in N(t_{n})} {D(x_{1},x_{2}; t_{n})}$

The primary tuning parameters for adaptivity are the history parameter $\Lambda$, the coarsening constant $C_c$, and the refining constant $C_r$. Their effects can be interpreted as:

- Higher values of the history parameter $$ \Lambda $$ imply lower significance of the similarity measures in the previous timestep on the similarity measure and thus adaptivity state in the current timestep.
- Higher values of the coarsening constant $$ C_c $$ imply that more active simulations from the previous timestep will remain active in the current timestep.
- Higher values of the refining constant $$ C_r $$ imply that less inactive points from the previous timestep will become active in the current timestep.

See the [adaptivity configuration](tooling-micro-manager-configuration.html#adaptivity) documentation on how to configure the parameters $C_c$, $C_r$, and more.

## Adaptivity variants

If the Micro Manager is run in parallel, micro simulations are distributed over MPI ranks. This opens the door different ways in which the adaptivity can be computed. There are two principle ways to go about it.

### Local adaptivity

Simulations on one rank are compared against each other. The similarity distance matrix $D$ has size $[N_l,N_l]$ on every rank, where $N_l$ is the local number of simulations. Each rank consecutively computes its own similarity distance matrix, and decides its set of active and inactive simulations.

### Global adaptivity

Each simulation is compared to all other simulations in the global domain. The similarity distance matrix $D$ has size $[N_g,N_g]$ on every compute node, where $N_g$ is the global number of simulations. Note that one copy of the similarity distance matrix $D$ is stored on every compute node, and **not** on every rank. We use MPI-based shared memory storage and access to store and update only one copy of the $D$ matrix on every node. The local primary rank (lowest rank on every node) updates the $D$ matrix. This implementation enables some memory saving for large cases with global adaptivity.
