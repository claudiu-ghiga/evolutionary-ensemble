import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import jaccard, cosine, euclidean, cdist
from scipy.stats import mode
from scipy.stats.stats import _compute_dminus
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_blobs, make_gaussian_quantiles
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.validation import check_scalar
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils import print_percent_done
from datasets import TestDatagen


class SVMFitness:
    """ Binary SVM optimization criteria. """
    def __init__(self, _X, _y, **kwargs):
        self._X = _X
        self._y = _y
        self._C = kwargs["_C"] if "_C" in kwargs else 1.0

    def __call__(self, *args):
        w = args[0]
        w_sqnorm = 0.5 * np.linalg.norm(w, 2)
        loss = w_sqnorm + self.get_C() * np.sum(
            np.maximum(0, 1 - self.get_y() * np.dot(
                w,
                self.get_X().T
            ))
        )
        return -loss

    def get_X(self):
        return self._X

    def get_y(self):
        return self._y

    def get_C(self):
        return self._C

class LDAFitness:
    """ Binary Fisher LDA optimization criteria. """
    def __init__(self, _X, _y, **kwargs):
        self._X = _X
        self._y = _y
        self.X0 = _X[_y == -1]
        self.X1 = _X[_y == 1]
        self.mu0 = np.mean(self.X0, axis=0)
        self.mu1 = np.mean(self.X1, axis=0)
        self.sigma0 = np.cov(self.X0.T)
        self.sigma1 = np.cov(self.X1.T)

    def __call__(self, *args):
        """*args = W, w"""
        w = args[0]
        Sb = np.dot(w, self.mu1 - self.mu0) ** 2
        Sw = np.dot(np.dot(w, self.sigma0 + self.sigma1), w.T)
        return Sb / Sw

    def get_X(self):
        return self._X

    def get_y(self):
        return self._y

class EuclideanFitness:
    def __init__(self, _X, _y, **kwargs):
        self._X = _X
        self._y = _y

    def __call__(self, *args):
        """*args = W, w"""
        w, W = args[0], args[1]
        dmatrix = cdist(W, w.reshape(1, -1), 'euclidean')
        return np.mean(dmatrix, axis=0)[0]

    def get_X(self):
        return self._X

    def get_y(self):
        return self._y

class CosineFitness:
    def __init__(self, _X, _y, **kwargs):
        self._X = _X
        self._y = _y

    def __call__(self, *args):
        """*args = W, w"""
        w, W = args[0], args[1]
        dmatrix = cdist(W, w.reshape(1, -1), 'cosine')
        return np.mean(dmatrix, axis=0)[0]

    def get_X(self):
        return self._X

    def get_y(self):
        return self._y

class BiObjectiveEuclid:
    def __init__(self, _X, _y, **kwargs):
        self._X = _X
        self._y = _y
        self._svm = SVMFitness(_X, _y, **kwargs)
        self._euclid = EuclideanFitness(_X, _y, **kwargs)
        self.n_obj = 2

    def __call__(self, *args):
        indiv = args[0]
        pop = args[1]
        return np.r_[self._svm(indiv), self._euclid(indiv, pop)]

class BiObjectiveCosine:
    def __init__(self, _X, _y, **kwargs):
        self._X = _X
        self._y = _y
        self._svm = SVMFitness(_X, _y, **kwargs)
        self._cosine = CosineFitness(_X, _y, **kwargs)
        self.n_obj = 2

    def __call__(self, *args):
        indiv = args[0]
        pop = args[1]
        return np.r_[self._svm(indiv), self._cosine(indiv, pop)]

class BiObjectiveLDA:
    def __init__(self, _X, _y, **kwargs):
        self._X = _X
        self._y = _y
        self._svm = SVMFitness(_X, _y, **kwargs)
        self._lda = LDAFitness(_X, _y, **kwargs)
        self.n_obj = 2

    def __call__(self, *args):
        indiv = args[0]
        return np.r_[self._svm(indiv), self._lda(indiv)]

class CDE(BaseEstimator, ClassifierMixin):
    """
    Crowding Differential Evolution Ensemble Learner.

    Args:
        fobj: objective function
        f: scale factor with values between [0, 2]
        cr: crossover rate with values between [0, 1]
        ps: population size
        max_iter: number of iterations for DE
    """
    def __init__(self, fobj_constr=SVMFitness, bounds=[(-1.0, 1.0)],
                 f=0.8, cr=0.7, ps=100, max_iter=100,
                 viz2d=False, crowding_metric=None,
                 run_id=None, C=1.0):
        self.fobj_constr = fobj_constr
        self.bounds = bounds
        self.f = f
        self.cr = cr
        self.ps = ps
        self.max_iter = max_iter
        self.viz2d = viz2d
        self.crowding_metric = crowding_metric
        self.run_id = run_id
        self.C = C

    def fit(self, X, y):
        # Validate parameters.
        check_scalar(self.f, "F", float, min_val=0.0, max_val=2.0)
        check_scalar(self.cr, "CR", float, min_val=0.0, max_val=1.0)
        check_scalar(self.ps, "PS", int, min_val=3)
        check_scalar(self.max_iter, "max_iter", int, min_val=0)
        check_X_y(X, y)
        # Store training data as attributes.
        self.X_ = X
        self.y_ = y
        self.bounds = self.bounds * X.shape[1]
        # Store classes.
        self.classes_ = np.unique(y)
        if self.classes_.size > 2:
            raise ValueError("Use a classifier from sklearn.multiclass if \
            you have more than two labels.")
        if (idxs := np.where(self.y_ == 0)[0]) is not None:
            self.y_[idxs] = -1
            self.classes_[self.classes_ == 0] = -1
        self.fobj = self.fobj_constr(self.X_, self.y_, _C=self.C)
        # Initialize differential evolution.
        dims = len(self.bounds)
        pop = np.random.rand(self.ps, dims)
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([self.fobj(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        self.pop_ = pop
        best = pop_denorm[best_idx]
        for i in range(self.max_iter):
            print_percent_done(i, self.max_iter, title=f"Run {self.run_id}")
            for j in range(self.ps):
                idxs = [idx for idx in range(self.ps) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), 0, 1)
                cross_points = np.random.rand(dims) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dims)] = True
                trial = np.where(cross_points, mutant, pop[j])
                trial_denorm = min_b + trial * diff
                trial_fit = self.fobj(trial_denorm)
                r_idx = self._get_replace_idx(j, trial)
                if trial_fit > fitness[r_idx]:
                    fitness[r_idx] = trial_fit
                    pop[r_idx] = trial
                    if trial_fit > fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
            self.pop_ = pop
            # print(self.predict(self.X_))
            # print("Best:", fitness[best_idx])
            # print("Mean:", np.mean(fitness))
        return self

    def predict(self, X):
        check_is_fitted(self)
        check_array(X)
        n_obs = X.shape[0]
        n_pop = self.pop_.shape[0]
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + self.pop_ * diff
        y_pred = np.where(np.dot(pop_denorm, X.T) < 0,
                          -np.ones((n_pop, n_obs)),
                          np.ones((n_pop, n_obs))).astype(int)
        m = mode(y_pred)
        return m[0][0]

    def decision_function(self, X):
        check_is_fitted(self)
        check_array(X)
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + self.pop_ * diff
        return np.mean(np.dot(pop_denorm, X.T), axis=0)

    def base_learner_predict(self, X):
        check_is_fitted(self)
        check_array(X)
        n_obs = X.shape[0]
        n_pop = self.pop_.shape[0]
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + self.pop_ * diff
        return np.where(np.dot(pop_denorm, X.T) < 0,
                        -np.ones((n_pop, n_obs)),
                        np.ones((n_pop, n_obs))).astype(int)

    def _predict_one(self, X, w):
        n_obs = X.shape[0]
        return np.where(np.dot(w, X.T) < 0,
                        -np.ones(n_obs),
                        np.ones(n_obs))

    def _get_replace_idx(self, idx, trial):
        if self.crowding_metric is None:
            return idx
        else:
            # Compute crowding distances.
            v = trial.reshape(1, -1)
            distances = cdist(self.pop_, v, self.crowding_metric).reshape(-1)
            return np.argmin(distances)


class DEMO(BaseEstimator, ClassifierMixin):
    """
    Multiobjective Differential Evolution Ensemble Learner.

    Args:
        fobj: function that returns a list of objectives
        f: scale factor with values between [0, 2]
        cr: crossover rate with values between [0, 1]
        ps: population size
        max_iter: number of iterations for DE
    """
    def __init__(self, fobj_constr=BiObjectiveLDA, bounds=[(-1.0, 1.0)], f=0.8, cr=0.7,
                 ps=100, max_iter=100, run_id=None,
                 C=1.0):
        self.fobj_constr = fobj_constr
        self.bounds = bounds
        self.f = f
        self.cr = cr
        self.ps = ps
        self.max_iter = max_iter
        self.run_id = run_id
        self.C = C

    def fit(self, X, y):
        check_scalar(self.f, "F", float, min_val=0.0, max_val=2.0)
        check_scalar(self.cr, "CR", float, min_val=0.0, max_val=1.0)
        check_scalar(self.ps, "PS", int, min_val=3)
        check_scalar(self.max_iter, "max_iter", int, min_val=0)
        check_X_y(X, y)
        # Store training data as attributes.
        self.X_ = X
        self.y_ = y
        self.bounds = self.bounds * X.shape[1]
        self.classes_ = np.unique(y)
        if self.classes_.size > 2:
            raise ValueError("Use a classifier from sklearn.multiclass if \
            you have more than two labels.")
        if (idxs := np.where(self.y_ == 0)[0]) is not None:
            self.y_[idxs] = -1
            self.classes_[self.classes_ == 0] = -1
        self.fobj = self.fobj_constr(self.X_, self.y_, _C=self.C)
        # Initialize differential evolution.
        dims = len(self.bounds)
        pop = np.random.rand(self.ps, dims)
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        self.pop_ = pop
        # Evaluate population.
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([self.fobj(ind, pop_denorm)
                              for ind in pop_denorm])
        for i in range(self.max_iter):
            print(f"Iteration {i}.")
            candidate_pop = []
            candidate_fits = []
            for j in range(self.ps):
                idxs = [idx for idx in range(self.ps) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), 0, 1)
                cross_points = np.random.rand(dims) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dims)] = True
                trial = np.where(cross_points, mutant, pop[j])
                trial_denorm = min_b + trial * diff
                trial_fit = self.fobj(trial_denorm, pop_denorm)
                if self._dominates(trial_fit, fitness[j]):
                    pop[j] = trial
                elif self._dominates(fitness[j], trial_fit):
                    continue
                else:
                    candidate_pop.append(trial)
                    candidate_fits.append(trial_fit)
            if len(candidate_pop) > 1:
                candidate_pop = np.asarray(candidate_pop)
                candidate_fits = np.asarray(candidate_fits)
                # Truncate population.
                pop, fitness = self._truncate(np.r_[pop, candidate_pop],
                                              np.r_[fitness, candidate_fits],
                                              self.ps, return_fitness=True)
            # print("global mean")
            print(np.mean(fitness, axis=0))
            # time.sleep(5)
        fronts = self._non_dominated_sort(fitness)
        self.pop_ = pop
        self.fronts_ = fronts
        self.fitness_ = fitness

    def predict(self, X):
        check_is_fitted(self)
        check_array(X)
        n_obs = X.shape[0]
        n_pop = self.pop_.shape[0]
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + self.pop_ * diff
        y_pred = np.where(np.dot(pop_denorm, X.T) < 0,
                          -np.ones((n_pop, n_obs)),
                          np.ones((n_pop, n_obs)))
        m = mode(y_pred)
        return m[0][0]

    def decision_function(self, X):
        check_is_fitted(self)
        check_array(X)
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + self.pop_ * diff
        return np.mean(np.dot(pop_denorm, X.T), axis=0)

    def base_learner_predict(self, X):
        check_is_fitted(self)
        check_array(X)
        n_obs = X.shape[0]
        n_pop = self.pop_.shape[0]
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + self.pop_ * diff
        return np.where(np.dot(pop_denorm, X.T) < 0,
                        -np.ones((n_pop, n_obs)),
                        np.ones((n_pop, n_obs))).astype(int)

    def _predict_one(self, X, w):
        n_obs = X.shape[0]
        return np.where(np.dot(w, X.T) < 0,
                        -np.ones(n_obs),
                        np.ones(n_obs))

    def _crowding_distance_metric(self, fits):
        l, m = fits.shape
        dists = np.zeros(l)
        sorted_idxs = np.argsort(fits, axis=0)
        for i in range(m):
            sorted_fits = fits[sorted_idxs[:, i], i]
            dists[sorted_idxs[0]] = dists[sorted_idxs[l-1]] = np.inf
            for j in range(1, l-1):
                dists[sorted_idxs[j]] += sorted_fits[j+1] - sorted_fits[j-1]
                # Alternative distance:
                # dists[sorted_idxs[j]] += sorted_fits[j+1] - sorted_fits[j]
        return dists

    def _dominates(self, fitA, fitB):
        assert fitA.size == fitB.size
        # Does A dominate B?
        dominates = False
        for i in range(fitA.size):
            if fitA[i] > fitB[i]:
                dominates = True
            elif fitB[i] > fitA[i]:
                return False
        return dominates

    def _non_dominated_sort(self, fits):
        S = [[] for _ in range(fits.shape[0])]
        Fs = []
        F = []
        n = [0 for _ in range(fits.shape[0])]
        for p in range(fits.shape[0]):
            for q in range(fits.shape[0]):
                if self._dominates(fits[p], fits[q]):
                    S[p].append(q)
                elif self._dominates(fits[q], fits[p]):
                    n[p] += 1
            if n[p] == 0:
                F.append(p)
        # print("fits:", fits)
        # print("F:", F)
        Fs.append(F)
        i = 0
        while Fs[i]:
            Fi = []
            for p in Fs[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        Fi.append(q)
            i += 1
            Fs.append(Fi)
        return Fs[:-1]

    def _truncate(self, pop, fits, pop_size, return_fitness=True):
        fronts = self._non_dominated_sort(fits)
        #########################################
        # print(fronts)
        # for idx, f in enumerate(fronts):
        #     print(f"Front {idx}")
        #     print("fitness values:")
        #     print(fits[f])
        #     print("means")
        #     print(np.mean(fits[f], axis=0))
        #########################################
        n_front = curr_front = 0
        while n_front + len(fronts[curr_front]) <= pop_size:
            n_front += len(fronts[curr_front])
            curr_front += 1
        crowding_dists = self._crowding_distance_metric(fits[fronts[curr_front]])
        from_fronts = [i for f in fronts[:curr_front] for i in f]
        remaining = np.argsort(crowding_dists)[::-1][:pop_size - n_front]
        idxs = np.r_[from_fronts, remaining].astype(int)
        if return_fitness:
            return pop[idxs], fits[idxs]
        return pop[idxs]


class Stacking(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass


class PretrainedHyperplanes(BaseEstimator, ClassifierMixin):
    def __init__(self, W):
        self.W = W

    def fit(self, X, y):
        pass

    def predict(self, X):
        n_obs, n_feat = X.shape
        return np.where(np.dot(self.w, X.T) < 0,
                        -np.ones(n_obs),
                        np.ones(n_obs))

class CCEL(BaseEstimator, ClassifierMixin):
    """
    Differential Evolution Ensemble Learner with
    Cooperative Coevolution.

    Args:
        fobj: function that returns a list of objectives
        f: scale factor with values between [0, 2]
        cr: crossover rate with values between [0, 1]
        ps: population size
        max_iter: number of iterations for DE
        T: number of base classifiers in ensemble
    """
    def __init__(self, fobj_constr=SVMFitness, bounds=[(-1.0, 1.0)], f=0.8, cr=0.7,
                 ps=100, max_iter=100, run_id=None,
                 C=1.0, T=7):
        self.fobj_constr = fobj_constr
        self.bounds = bounds
        self.f = f
        self.cr = cr
        # Size of a single subpopulation.
        self.ps = ps
        self.pops = []
        self.max_iter = max_iter
        self.run_id = run_id
        self.C = C
        self.T = T

    def fit(self, X, y):
        check_scalar(self.f, "F", float, min_val=0.0, max_val=2.0)
        check_scalar(self.cr, "CR", float, min_val=0.0, max_val=1.0)
        check_scalar(self.ps, "PS", int, min_val=3)
        check_scalar(self.max_iter, "max_iter", int, min_val=0)
        check_X_y(X, y)
        # Store training data as attributes.
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        if self.classes_.size > 2:
            raise ValueError("Use a classifier from sklearn.multiclass if \
            you have more than two labels.")
        if (idxs := np.where(self.y_ == 0)[0]) is not None:
            self.y_[idxs] = -1
            self.classes_[self.classes_ == 0] = -1
        self.fobj = self.fobj_constr(self.X_, self.y_, _C=self.C)
        self.bounds = self.bounds * X.shape[1]
        self.dims = len(self.bounds)
        self.min_b, self.max_b = np.asarray(self.bounds).T
        self.diff = np.fabs(self.min_b - self.max_b)
        # Initialize T subpopulations and fitness.
        pops, fits, bests = self._init_pops()
        # For all generations...
        for i in range(self.max_iter):
            # print(f"Iteration {i}.")
            # print("Fitness:")
            # print(fits.reshape(self.T, -1).mean(axis=1))
            pop_idxs = np.arange(self.ps)
            # For all populations...
            for j in range(self.T):
                pop = pops[pop_idxs]
                new_pop = np.empty(pop.shape)
                fit = fits[pop_idxs]
                for k in range(self.ps):
                    idxs = [idx for idx in range(self.ps) if idx != k]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.f * (b - c), 0, 1)
                    cross_points = np.random.rand(self.dims) < self.cr
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dims)] = True
                    trial = np.where(cross_points, mutant, pop[k])
                    trial_fit = self._fitness(trial, j, bests)
                    if trial_fit > self._bests_fitness(bests):
                        bests[j] = trial
                    if trial_fit > fit[k]:
                        new_pop[k] = trial
                        fit[k] = trial_fit
                    else:
                        new_pop[k] = pop[k]
                fits[pop_idxs] = fit[:]
                pops[pop_idxs] = new_pop[:, :]
                pop_idxs = pop_idxs + self.ps
            # print("Subpopulation fitness:")
            # print(fits.reshape(self.T, -1).mean(axis=1))
            # print(f"Ensemble accuracy (train): {self._acc(bests)}")
        self.ensemble_ = bests
        self.pops_ = pops
        self.fitness_ = fits

    def predict(self, X):
        check_is_fitted(self)
        check_array(X)
        preds = np.where(
            np.dot(self._denorm(self.ensemble_), X.T) < 0, -1, 1
        ).astype(int)
        majority_vote = mode(preds)[0][0]
        return majority_vote

    def decision_function(self, X):
        check_is_fitted(self)
        check_array(X)
        return np.mean(
            np.dot(self._denorm(self.ensemble_), X.T),
            axis=0
        )

    def base_learner_predict(self, X):
        check_is_fitted(self)
        check_array(X)
        return np.where(
            np.dot(self._denorm(self.ensemble_), X.T) < 0, -1, 1
        ).astype(int)

    def _acc(self, b):
        preds = np.where(np.dot(self._denorm(b), self.X_.T) < 0, -1, 1)
        majority_vote = mode(preds)[0][0]
        return np.mean(majority_vote == self.y_)

    def _init_pops(self):
        # Initialize b vector.
        b = np.random.rand(self.T, self.dims)
        # Initialize subpopulations.
        pops = np.random.rand(self.T * self.ps, self.dims)
        # Evaluate fitness for each subpopulation.
        idxs = np.arange(self.ps)
        fits = np.empty(self.ps * self.T)
        for i in range(self.T):
            pop = pops[idxs, :]
            # Replace position i in b with each member of subpopulation i.
            ensembles = np.tile(b, (self.ps, 1, 1))
            ensembles[:, i, :] = pop
            subpop_fits = self._init_fitness(ensembles)
            fits[idxs] = subpop_fits
            idxs = idxs + self.ps
        return pops, fits, b

    def _fitness(self, o, pop_idx, b):
        ensemble = b[:]
        ensemble[pop_idx] = o[:]
        return np.mean([self.fobj(ind) for ind in self._denorm(ensemble)],
                       axis=0)

    def _bests_fitness(self, b):
        return np.mean([self.fobj(ind) for ind in self._denorm(b)],
                       axis=0)

    def _init_fitness(self, ensembles):
        return np.array(
            [np.mean([self.fobj(ind) for ind in self._denorm(ensemble)], axis=0)
             for ensemble in ensembles]
        )

    def _denorm(self, pop):
        return self.min_b + pop * self.diff

class CCMOEL(BaseEstimator, ClassifierMixin):
    """
    Multiobjective Differential Evolution Ensemble Learner with
    Cooperative Coevolution.

    Args:
        fobj: function that returns a list of objectives
        f: scale factor with values between [0, 2]
        cr: crossover rate with values between [0, 1]
        ps: population size
        max_iter: number of iterations for DE
        T: number of base classifiers in ensemble
    """
    def __init__(self, fobj_constr=SVMFitness, bounds=[(-1.0, 1.0)], f=0.8, cr=0.7,
                 ps=100, max_iter=100, run_id=None,
                 C=1.0, T=7):
        self.fobj_constr = fobj_constr
        self.bounds = bounds
        self.f = f
        self.cr = cr
        # Size of a single subpopulation.
        self.ps = ps
        self.pops = []
        self.max_iter = max_iter
        self.run_id = run_id
        self.C = C
        self.T = T

    def fit(self, X, y):
        check_scalar(self.f, "F", float, min_val=0.0, max_val=2.0)
        check_scalar(self.cr, "CR", float, min_val=0.0, max_val=1.0)
        check_scalar(self.ps, "PS", int, min_val=3)
        check_scalar(self.max_iter, "max_iter", int, min_val=0)
        check_X_y(X, y)
        # Store training data as attributes.
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        if self.classes_.size > 2:
            raise ValueError("Use a classifier from sklearn.multiclass if \
            you have more than two labels.")
        if (idxs := np.where(self.y_ == 0)[0]) is not None:
            self.y_[idxs] = -1
            self.classes_[self.classes_ == 0] = -1
        self.fobj = self.fobj_constr(self.X_, self.y_, _C=self.C)
        self.bounds = self.bounds * X.shape[1]
        self.dims = len(self.bounds)
        self.min_b, self.max_b = np.asarray(self.bounds).T
        self.diff = np.fabs(self.min_b - self.max_b)
        # Initialize T subpopulations and fitness.
        pops, fits, bests, first_fronts = self._init_pops()
        # For all generations...
        for i in range(self.max_iter):
            print(f"Iteration {i}.")
            pop_idxs = np.arange(self.ps)
            # For all populations...
            for j in range(self.T):
                pop = pops[pop_idxs]
                fit = fits[pop_idxs]
                collabs = self._generate_collaborators(first_fronts)
                candidate_pop = []
                candidate_fits = []
                for k in range(self.ps):
                    idxs = [idx for idx in range(self.ps) if idx != k]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.f * (b - c), 0, 1)
                    cross_points = np.random.rand(self.dims) < self.cr
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dims)] = True
                    trial = np.where(cross_points, mutant, pop[k])
                    trial_fit = self._fitness(trial, j, collabs)
                    if self._dominates(trial_fit, self._bests_fitness(collabs)):
                        bests[j] = trial
                    if self._dominates(trial_fit, fit[k]):
                        pop[k] = trial
                        fit[k] = trial_fit
                    elif self._dominates(fit[k], trial_fit):
                        continue
                    else:
                        candidate_pop.append(trial)
                        candidate_fits.append(trial_fit)
                if len(candidate_pop) > 1:
                    candidate_pop = np.asarray(candidate_pop)
                    candidate_fits = np.asarray(candidate_fits)
                    pop, fitness = self._truncate(np.r_[pop, candidate_pop],
                                                  np.r_[fit, candidate_fits],
                                                  self.ps, return_fitness=True)
                fits[pop_idxs] = fit[:]
                pops[pop_idxs] = pop[:, :]
                pop_idxs = pop_idxs + self.ps
            # print("Subpopulation fitness:")
            # print(fits.reshape(self.T, -1).mean(axis=1))
            # print(f"Ensemble accuracy (train): {self._acc(bests)}")
        self.ensemble_ = bests
        self.pops_ = pops
        self.fitness_ = fits

    def predict(self, X):
        check_is_fitted(self)
        check_array(X)
        preds = np.where(
            np.dot(self._denorm(self.ensemble_), X.T) < 0, -1, 1
        ).astype(int)
        majority_vote = mode(preds)[0][0]
        return majority_vote

    def decision_function(self, X):
        check_is_fitted(self)
        check_array(X)
        return np.mean(
            np.dot(self._denorm(self.ensemble_), X.T),
            axis=0
        )

    def base_learner_predict(self, X):
        check_is_fitted(self)
        check_array(X)
        return np.where(
            np.dot(self._denorm(self.ensemble_), X.T) < 0, -1, 1
        ).astype(int)

    def _crowding_distance_metric(self, fits):
        l, m = fits.shape
        dists = np.zeros(l)
        sorted_idxs = np.argsort(fits, axis=0)
        for i in range(m):
            sorted_fits = fits[sorted_idxs[:, i], i]
            dists[sorted_idxs[0]] = dists[sorted_idxs[l-1]] = np.inf
            for j in range(1, l-1):
                dists[sorted_idxs[j]] += sorted_fits[j+1] - sorted_fits[j-1]
                # Alternative distance:
                # dists[sorted_idxs[j]] += sorted_fits[j+1] - sorted_fits[j]
        return dists

    def _dominates(self, fitA, fitB):
        assert fitA.size == fitB.size
        # Does A dominate B?
        dominates = False
        for i in range(fitA.size):
            if fitA[i] > fitB[i]:
                dominates = True
            elif fitB[i] > fitA[i]:
                return False
        return dominates

    def _non_dominated_sort(self, fits):
        S = [[] for _ in range(fits.shape[0])]
        Fs = []
        F = []
        n = [0 for _ in range(fits.shape[0])]
        for p in range(fits.shape[0]):
            for q in range(fits.shape[0]):
                if self._dominates(fits[p], fits[q]):
                    S[p].append(q)
                elif self._dominates(fits[q], fits[p]):
                    n[p] += 1
            if n[p] == 0:
                F.append(p)
        # print("fits:", fits)
        # print("F:", F)
        Fs.append(F)
        i = 0
        while Fs[i]:
            Fi = []
            for p in Fs[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        Fi.append(q)
            i += 1
            Fs.append(Fi)
        return Fs[:-1]

    def _truncate(self, pop, fits, pop_size, return_fitness=True):
        fronts = self._non_dominated_sort(fits)
        #########################################
        # print(fronts)
        # for idx, f in enumerate(fronts):
        #     print(f"Front {idx}")
        #     print("fitness values:")
        #     print(fits[f])
        #     print("means")
        #     print(np.mean(fits[f], axis=0))
        #########################################
        n_front = curr_front = 0
        while n_front + len(fronts[curr_front]) <= pop_size:
            n_front += len(fronts[curr_front])
            curr_front += 1
        crowding_dists = self._crowding_distance_metric(fits[fronts[curr_front]])
        from_fronts = [i for f in fronts[:curr_front] for i in f]
        remaining = np.argsort(crowding_dists)[::-1][:pop_size - n_front]
        idxs = np.r_[from_fronts, remaining].astype(int)
        if return_fitness:
            return pop[idxs], fits[idxs]
        return pop[idxs]

    def _generate_collaborators(self, fronts):
        sizes = np.array([len(f) for f in fronts])
        choices = np.random.randint(sizes)
        return np.array([fronts[i][choices[i]] for i in range(len(choices))])

    def _acc(self, b):
        preds = np.where(np.dot(self._denorm(b), self.X_.T) < 0, -1, 1)
        majority_vote = mode(preds)[0][0]
        return np.mean(majority_vote == self.y_)

    def _init_pops(self):
        # Initialize b vector.
        b = np.random.rand(self.T, self.dims)
        # Initialize subpopulations.
        pops = np.random.rand(self.T * self.ps, self.dims)
        # Evaluate fitness for each subpopulation.
        idxs = np.arange(self.ps)
        fits = np.empty((self.ps * self.T, self.fobj.n_obj))
        first_fronts = []
        for i in range(self.T):
            pop = pops[idxs, :]
            # Replace position i in b with each member of subpopulation i.
            ensembles = np.tile(b, (self.ps, 1, 1))
            ensembles[:, i, :] = pop
            subpop_fits = self._init_fitness(ensembles)
            fits[idxs] = subpop_fits
            first_front = self._non_dominated_sort(subpop_fits)[0]
            first_fronts.append(pop[first_front])
            idxs = idxs + self.ps
        return pops, fits, b, first_fronts

    def _fitness(self, o, pop_idx, b):
        ensemble = b[:]
        ensemble[pop_idx] = o[:]
        return np.mean([self.fobj(ind) for ind in self._denorm(ensemble)],
                       axis=0)

    def _bests_fitness(self, b):
        return np.mean([self.fobj(ind) for ind in self._denorm(b)],
                       axis=0)

    def _init_fitness(self, ensembles):
        return np.array(
            [np.mean([self.fobj(ind) for ind in self._denorm(ensemble)], axis=0)
             for ensemble in ensembles]
        )

    def _denorm(self, pop):
        return self.min_b + pop * self.diff


def test_evolboost():
    def make_toy_dataset(n: int = 100, random_seed: int = None):
        """ Generate a toy dataset for evaluating AdaBoost classifiers """

        n_per_class = int(n/2)

        if random_seed:
            np.random.seed(random_seed)

        X, y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=2)

        return X, y*2-1

    def init_pop(n: int = 10, random_seed: int = None):
        if random_seed:
            np.random.seed(random_seed)

        hi, lo = [-1, 1]
        pop = lo + np.random.rand(n, 2) * (hi - lo)
        return pop

    def successes(X_, y_, pop_):
        y_preds = np.dot(pop_, X_.T)
        y_preds = np.where(y_preds < 0, -1, 1).astype(int)
        success = np.sum(y_preds == y_, axis=1) / y_preds.shape[1]
        return success

    def fitness(X_, y_, pop_, D_):
        y_preds = np.where(np.dot(pop_, X_.T) < 0, -1, 1).astype(int)
        return np.sum((y_preds == y_) * D_.reshape(1, -1), axis=1)

    def preds(X_, y_, pop_):
        return np.where(np.dot(pop_, X_.T) < 0, -1, 1).astype(int)

    def pred_ensemble(X_, y_, pop_, weights_=None):
        if weights_ is None:
            weights_ = np.ones(pop_.shape[0])
        print("pop_")
        print(pop_)
        print("X_^T")
        print(X_.T)
        margins = np.dot(pop_, X_.T)
        print("margins")
        print(margins)
        w_margins = np.sum(margins * weights_.reshape(-1, 1), axis=0)
        print("combined margins")
        print(w_margins)
        return np.where(w_margins < 0, -1, 1).astype(int)

    def find_classifier(fits_, ensemble_):
        best_ = np.argwhere(np.isclose(fits_, np.amax(fits_)))
        clf = None
        for idx in best_.flatten():
            if idx not in ensemble_:
                clf = idx
                break
        return clf

    n_obs = 500
    n_pop = 200
    T = 1
    X, y = make_toy_dataset(n=n_obs)
    pop = init_pop(n_pop)
    hits = successes(X, y, pop)
    ensemble = set()
    confidences = np.zeros(T)
    D = np.ones(n_obs) / X.shape[0]
    print(f"Base classifier accuracies: {hits}")
    print(f"Mean classifier accuracy: {np.mean(hits)}")
    t = 0
    while t < T:
        fits = fitness(X, y, pop, D)
        best = find_classifier(fits, ensemble)
        if best is None:
            print(f"Stopped prematurely at iteration {t} because",
                  "there are no more classifiers to choose from.")
            break
        else:
            ensemble.add(best)
        losses = np.fabs(np.where(np.dot(pop[best], X.T) < 0, -1, 1) - y) / 2.0
        avg_loss = np.sum(losses * D)
        beta = np.log(avg_loss / (1 - avg_loss))
        confidences[t] = beta
        new_D = D * (beta ** (1 - losses))
        D = new_D / np.sum(new_D)
        t += 1
    geom_median = 0.5 * np.sum(1 / confidences[:t])
    ensemble_w = pop[sorted(ensemble)]
    y_selected = preds(X, y, ensemble_w)
    y_ensemble = mode(y_selected)[0][0]
    y_ensemble_2 = pred_ensemble(X, y, ensemble_w, weights_=confidences[:t])
    print(f"Ensemble accuracy (majority voting): {np.sum(y_ensemble == y) / y.size}")
    print(f"Ensemble accuracy (weighted sum): {np.sum(y_ensemble_2 == y) / y.size}")


def main():
    pass
    # (X_train, y_train), (X_test, y_test) = TestDatagen(n_samples=500, n_features=5)()
    # clf = CCEL(fobj=SVMFitness(X_train, y_train), ps=10, T=9,
    #            max_iter=200)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print(f"Ensemble accuracy (test): {np.mean(y_pred == y_test)}")


if __name__ == "__main__":
    main()
