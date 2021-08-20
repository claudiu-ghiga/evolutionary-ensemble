import argparse
from collections import defaultdict
from statistics import mean, stdev
import sys
import timeit

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, auc
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC

from datasets import *
from algorithms import CDE, DEMO, BiObjectiveEuclid, BiObjectiveCosine, SVMFitness
from algorithms import BiObjectiveLDA, LDAFitness, CCEL, CCMOEL
from extensions import pairwise_errors

np.set_printoptions(threshold=1000)

class OneVRest(OneVsRestClassifier):
    def base_learner_predict(self, X):
        if (isinstance(self.estimator, DEMO) or
            isinstance(self.estimator, CDE)):
            return self.estimators_[0].base_learner_predict(X)

def diversity(Y_preds, y_true):
    metrics = dict()
    Y_errs = (Y_preds == y_true).astype(int)
    Q, D = pairwise_errors(Y_errs)
    print("DIVERSITY REPORT")
    print("-"*60)
    print("Q statistic")
    Q_triu_idxs = np.triu_indices(Q.shape[0], 1)
    sorted_Q_idxs = np.argsort(np.fabs(Q[Q_triu_idxs]))
    sorted_Q_pairs = np.array(Q_triu_idxs).T[sorted_Q_idxs]
    sorted_Q = Q[Q_triu_idxs][sorted_Q_idxs]
    print("Top 10 classifiers pairs by diversity:")
    for i in range(10):
        print(f"Pair {sorted_Q_pairs[i]}: {sorted_Q[i]}")
    print("Average Q statistic:", end=" ")
    print(np.average(Q[Q_triu_idxs]))
    print("Disagreement measure")
    D_triu_idxs = np.triu_indices(D.shape[0], 1)
    sorted_D_idxs = np.argsort(D[D_triu_idxs])[::-1]
    sorted_D_pairs = np.array(D_triu_idxs).T[sorted_D_idxs]
    sorted_D = D[D_triu_idxs][sorted_D_idxs]
    print("Top 10 classifiers pairs by diversity:")
    for i in range(10):
        print(f"Pair {sorted_D_pairs[i]}: {sorted_D[i]}")
    L, N = Y_errs.shape
    l = np.sum(Y_errs, axis=0)
    # Entropy
    E = np.sum(np.minimum(l, L - l)) / (N * (L - np.ceil(L / 2)))
    print("Entropy:", E)
    # Kohavi-Wolpert variance
    KW = np.sum(l * (L - l)) / (N * L ** 2)
    print("Kohavi-Wolpert variance:", KW)
    # Interrater agreement (k)
    p_mean = np.sum(l) / (N * L)
    kappa = 1 - (KW * L) / ((L - 1) * p_mean * (1 - p_mean))
    print("Interrater agreement:", kappa)
    # Difficulty distribution
    survivor_ps, survivor_bins = np.histogram(l, 20, density=True,
                                              range=(0, L))
    print("Difficulty distribution")
    fig, ax = plt.subplots()
    ax.grid(color="black", linestyle="--")
    ax.hist(l / L, survivor_bins / L,
            color="limegreen", edgecolor="black")
    ax.set(xlabel=r"Difficulty ($\theta$)",
           ylabel="Counts")
    fig.savefig("difficulty.eps")
    plt.clf()
    # Generalized diversity
    failure_ps, failure_bins = np.histogram((L - l), L, density=True,
                                            range=(0, L))
    p1 = np.dot(failure_bins[1:] / L, failure_ps)
    p2 = np.dot((failure_bins[1:] / L) * (failure_bins[1:] - 1) / (L - 1),
                failure_ps)
    print("Generalized diversity:", 1 - p2 / p1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",
                        action="store_true")
    parser.add_argument("-n",
                        action="store", type=int, default=30)
    parser.add_argument("-p",
                        action="store", type=int, default=100)
    parser.add_argument("-c",
                        action="store", type=float, default=0.9)
    parser.add_argument("-f",
                        action="store", type=float, default=1.2)
    parser.add_argument("-d",
                        action="store_true")
    parser.add_argument("-a",
                        action="store", type=int, default=1,
                        choices=[1,2,3,4,5,6,7,8,9,10,11])
    parser.add_argument("-x",
                        action="store", type=int, default=100)
    parser.add_argument("-o",
                        action="store", type=str)
    parser.add_argument("--phase-portrait",
                        action="store_true")
    args = parser.parse_args()
    algos = [
        (LinearSVC, {
            "C": 1.0
        }),
        (LinearDiscriminantAnalysis, {
            "solver": "lsqr"
        }),
        (CDE, {
            "fobj_constr": SVMFitness, "f": args.f, "cr": args.c,
            "ps": args.p, "bounds": [(-1.0, 1.0)], "max_iter": args.x,
            "crowding_metric": None
        }),
        (CDE, {
            "fobj_constr": SVMFitness, "f": args.f, "cr": args.c,
            "ps": args.p, "bounds": [(-1.0, 1.0)], "max_iter": args.x,
            "crowding_metric": "euclidean"
        }),
        (CDE, {
            "fobj_constr": SVMFitness, "f": args.f, "cr": args.c,
            "ps": args.p, "bounds": [(-1.0, 1.0)], "max_iter": args.x,
            "crowding_metric": "cosine"
        }),
        (DEMO, {
            "fobj_constr": BiObjectiveEuclid, "f": args.f, "cr": args.c,
            "ps": args.p, "bounds": [(-1.0, 1.0)], "max_iter": args.x
        }),
        (DEMO, {
            "fobj_constr": BiObjectiveCosine, "f": args.f, "cr": args.c,
            "ps": args.p, "bounds": [(-1.0, 1.0)], "max_iter": args.x
        }),
        (DEMO, {
            "fobj_constr": BiObjectiveLDA, "f": args.f, "cr": args.c,
            "ps": args.p, "bounds": [(-1.0, 1.0)], "max_iter": args.x
        }),
        (CCEL, {
            "fobj_constr": SVMFitness, "f": args.f, "cr": args.c,
            "ps": args.p, "bounds": [(-1.0, 1.0)], "max_iter": args.x
        }),
        (CCEL, {
            "fobj_constr": LDAFitness, "f": args.f, "cr": args.c,
            "ps": args.p, "bounds": [(-1.0, 1.0)], "max_iter": args.x
        }),
        (CCMOEL, {
            "fobj_constr": BiObjectiveLDA, "f": args.f, "cr": args.c,
            "ps": args.p, "bounds": [(-1.0, 1.0)], "max_iter": args.x
        })
    ]
    logfile = open(args.o, "w") if args.o is not None else sys.stdout
    print_verbose = lambda s: print(s) if args.verbose else lambda s: None
    data_generators = [
        WineQuality(), Wisconsin(), Poker(), Magic(),
        Yeast4(), Segment0(), PageBlocks0(), Iris0(),
        Glass4(), Abalone19(), Coil2000(), Iris()
    ]
    for gen in data_generators:
        totals = defaultdict(lambda : [])
        print(gen.fname, file=logfile)
        print("#"*50, file=logfile)
        (X_train, y_train), (X_test, y_test) = gen(verbose=args.verbose)
        alg, params = algos[args.a - 1]
        if args.phase_portrait:
            out_dict = {
                "f": [],
                "cr": [],
                "train_acc": [],
                "test_acc": [],
                "f1_train": [],
                "f1_test": []
            }
            out_df = pd.DataFrame.from_dict(out_dict)
            f_ngrid = 10
            cr_ngrid = 10
            n_points = 4
            f_step = 1 / f_ngrid
            cr_step = 1 / cr_ngrid
            i = 1
            for cr_start in np.linspace(0, 1, f_ngrid, endpoint=False):
                for f_start in np.linspace(0, 1, cr_ngrid, endpoint=False):
                    print(f"Iteration {i}")
                    points = np.random.rand(n_points, 2)
                    points[:, 0] = points[:, 0] * f_step + f_start
                    points[:, 1] = points[:, 1] * cr_step + cr_start
                    print(points)
                    for p in points:
                        f, cr = p
                        params["f"], params["cr"] = f, cr
                        clf = OneVRest(
                            alg(**params)
                        )
                        clf.fit(X_train, y_train)
                        y_train_pred = clf.predict(X_train)
                        y_test_pred = clf.predict(X_test)
                        acc_train = accuracy_score(y_train, y_train_pred)
                        acc_test = accuracy_score(y_test, y_test_pred)
                        f1_train = f1_score(y_train, y_train_pred, zero_division=0, average="weighted")
                        f1_test = f1_score(y_test, y_test_pred, zero_division=0, average="weighted")
                        out_df.loc[len(out_df)] = [f, cr, acc_train, acc_test, f1_train, f1_test]
                    i += 1
            out_df.to_csv("phase.csv", index=False)
        elif args.d:
            clf = OneVRest(
                alg(**params)
            )
            if np.unique(y_train).size > 2:
                raise ValueError("Diversity report available only for binary classification")
            clf.fit(X_train, y_train)
            Y_preds = clf.base_learner_predict(X_train)
            diversity(Y_preds, y_train)
        else:
            for i in range(args.n):
                if alg is not LinearSVC and alg is not LinearDiscriminantAnalysis:
                    params["run_id"] = i+1
                clf = OneVRest(
                    alg(**params)
                )
                clf.fit(X_train, y_train)
                y_train_pred = clf.predict(X_train)
                y_test_pred = clf.predict(X_test)
                totals["acc_train"].append(accuracy_score(y_train, y_train_pred))
                totals["prec_train"].append(precision_score(y_train, y_train_pred, zero_division=0, average="weighted"))
                totals["rec_train"].append(recall_score(y_train, y_train_pred, zero_division=0, average="weighted"))
                totals["f1_train"].append(f1_score(y_train, y_train_pred, zero_division=0, average="weighted"))
                totals["acc_test"].append(accuracy_score(y_test, y_test_pred))
                totals["prec_test"].append(precision_score(y_test, y_test_pred, zero_division=0, average="weighted"))
                totals["rec_test"].append(recall_score(y_test, y_test_pred, zero_division=0, average="weighted"))
                totals["f1_test"].append(f1_score(y_test, y_test_pred, zero_division=0, average="weighted"))
            print(f"Train mean accuracy: {mean(totals['acc_train'])} ± {stdev(totals['acc_train'])}",
                file=logfile)
            print(f"Train mean precision: {mean(totals['prec_train'])} ± {stdev(totals['prec_train'])}",
                file=logfile)
            print(f"Train mean recall: {mean(totals['rec_train'])} ± {stdev(totals['rec_train'])}",
                file=logfile)
            print(f"Train mean F1: {mean(totals['f1_train'])} ± {stdev(totals['f1_train'])}",
                file=logfile)
            print(f"Test mean accuracy: {mean(totals['acc_test'])} ± {stdev(totals['acc_test'])}",
                file=logfile)
            print(f"Test mean precision: {mean(totals['prec_test'])} ± {stdev(totals['prec_test'])}",
                file=logfile)
            print(f"Test mean recall: {mean(totals['rec_test'])} ± {stdev(totals['rec_test'])}",
                file=logfile)
            print(f"Test mean F1: {mean(totals['f1_test'])} ± {stdev(totals['f1_test'])}",
                file=logfile)
            print("#"*50)


if __name__ == "__main__":
    main()
