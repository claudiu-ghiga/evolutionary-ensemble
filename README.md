# evolutionary-ensemble

Differential evolution algorithm that creates classifier ensembles. The optimization procedure is multiobjective and cooperative.

The multiobjective aspect of the algorithm is inspired by the DEMO procedure (https://doi.org/10.1007/978-3-540-31880-4_36). The cooperation between candidate solutions takes place at the evaluation step, where candidates from different subpopulations are chosen to take part in the ensemble and the fitness score is assigned to each individual based on its contribution to the predictive power of the total ensemble.
