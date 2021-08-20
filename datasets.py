import os
import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandas.core.series import Series
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler


class Dataset(ABC):
    def __init__(self, data_type, fname):
        self.data_root = "~/repos/evolutionary-classifiers/data"
        self.data_prefix = os.path.join(self.data_root, data_type)
        self.fname = fname

    @abstractmethod
    def _preprocess(self):
        pass

    def describe(self):
        n_obs, n_attrs = self.X.shape
        labels, counts = np.unique(self.y, return_counts=True)
        n_majority = np.max(counts)
        n_rest = np.sum(counts) - n_majority
        print(self.X.shape)
        print("# observations:", n_obs)
        # Remove bias.
        print("# attributes:", n_attrs)
        print("# classes:", labels.size)
        print("IR:", n_majority / n_rest)

    def __call__(self, train_frac=0.8, verbose=False):
        df = self._preprocess()
        # Train-test split.
        n_train = int(train_frac * len(df.index))
        self.X, self.y = (df.iloc[:, :-1].to_numpy(),
                          df.iloc[:, -1].to_numpy())
        X_train, X_test = self.X[:n_train], self.X[n_train:]
        y_train, y_test = self.y[:n_train], self.y[n_train:]
        # Describe dataset.
        if verbose:
            self.describe()
        # Scale features.
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Add bias term.
        X_train, X_test = (np.c_[X_train, np.ones(X_train.shape[0])],
                           np.c_[X_test, np.ones(X_test.shape[0])])
        return ((X_train, y_train), (X_test, y_test))


class TestDatagen(Dataset):
    def __init__(self, n_centers=2, n_samples=500, n_features=4,
                 cluster_std=3):
        self.fname = "Test dataset"
        self.n_centers = n_centers
        self.n_samples = n_samples
        self.n_features = n_features
        self.cluster_std = cluster_std

    def _preprocess(self):
        X, y = make_blobs(n_samples=self.n_samples, centers=self.n_centers,
                          n_features=self.n_features,
                          cluster_std=self.cluster_std)
        y = (np.where(y == 0, -1, 1)
             if self.n_centers == 2
             else y)
        df = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
        return df


class WineQuality(Dataset):
    def __init__(self):
        super().__init__("standard", "winequality")

    def _preprocess(self):
        paths = [
            "wine-quality/winequality-red.dat",
            "wine-quality/winequality-white.dat"
        ]
        columns = [
            "FixedAcidity", "VolatileAcidity", "CitricAcid",
            "ResidualSugar", "Chlorides", "FreeSulfurDioxide",
            "TotalSulfurDioxide", "Density", "PH", "Sulphates",
            "Alcohol", "Quality"
        ]
        red_df = pd.read_csv(os.path.join(self.data_prefix, paths[0]),
                            sep=",", skiprows=16, header=None)
        red_df.columns = columns
        white_df = pd.read_csv(os.path.join(self.data_prefix, paths[1]),
                            sep=",", skiprows=16, header=None)
        white_df.columns = columns
        red_df["Class"] = 1
        white_df["Class"] = -1
        wine_df = pd.concat([red_df, white_df]).sample(frac=1.0) \
                    .reset_index(drop=True)
        wine_df = wine_df.drop(columns=["Quality"])
        return wine_df


class Wisconsin(Dataset):
    def __init__(self):
        super().__init__("standard", "wisconsin")

    def _preprocess(self):
        path = "wisconsin/wisconsin.dat"
        columns = [
            "ClumpThickness", "CellSize", "CellShape",
            "MarginalAdhesion", "EpithelialSize", "BareNuclei",
            "BlandChromatin", "NormalNucleoli", "Mitoses", "Class"
        ]
        wisconsin_df = pd.read_csv(os.path.join(self.data_prefix, path),
                                sep=",", skiprows=14, header=None)
        wisconsin_df.columns = columns
        # print("Value: :::%s:::" % wisconsin_df.iloc[22]["BareNuclei"])
        # print("Null values?", wisconsin_df.isnull().values.any())
        wisconsin_df = wisconsin_df[wisconsin_df["BareNuclei"] != " <null>"]
        wisconsin_df["BareNuclei"] = pd.to_numeric(wisconsin_df["BareNuclei"])
        wisconsin_df = wisconsin_df.sample(frac=1.0).reset_index(drop=True)
        wisconsin_df["Class"] = np.where(wisconsin_df["Class"] == 2, -1, 1)
        return wisconsin_df


class Poker(Dataset):
    def __init__(self):
        super().__init__("multiclass", "poker")

    def _preprocess(self):
        path = "poker/poker.dat"
        columns = [
            "S1", "C1", "S2", "C2", "S3", "C3",
            "S4", "C4", "S5", "C5", "Class"
        ]
        poker_df = pd.read_csv(os.path.join(self.data_prefix, path),
                            sep=",", skiprows=15, header=None)
        poker_df.columns = columns
        poker_df = poker_df.sample(frac=0.05).reset_index(drop=True)
        return poker_df


class Magic(Dataset):
    def __init__(self):
        super().__init__("standard", "magic")

    def _preprocess(self):
        path = "magic/magic.dat"
        columns = [
            "FLength", "FWidth", "FSize", "FConc", "FConc1",
            "FAsym", "FM3Long", "FM3Trans", "FAlpha", "FDist",
            "Class"
        ]
        magic_df = pd.read_csv(os.path.join(self.data_prefix, path),
                               skiprows=15, sep=",", header=None)
        magic_df.columns = columns
        magic_df = magic_df.sample(frac=1.0).reset_index(drop=True)
        magic_df["Class"] = np.where(magic_df["Class"] == "g", 1, -1)
        return magic_df


class Yeast4(Dataset):
    def __init__(self):
        super().__init__("imbalanced", "yeast4")

    def _preprocess(self):
        path = "yeast4/yeast4.dat"
        columns = [
            "Mcg", "Gvh", "Alm", "Mit", "Erl", "Pox", "Vac",
            "Nuc", "Class"
        ]
        yeast_df = pd.read_csv(os.path.join(self.data_prefix, path),
                               skiprows=13, sep=",", header=None)
        yeast_df.columns = columns
        yeast_df = yeast_df.sample(frac=1.0).reset_index(drop=True)
        yeast_df["Class"] = np.where(yeast_df["Class"] == " negative", -1, 1)
        return yeast_df


class Segment0(Dataset):
    def __init__(self):
        super().__init__("imbalanced", "segment0")

    def _preprocess(self):
        path = "segment0/segment0.dat"
        columns = [
            "Region-centroid-col", "Region-centroid-row",
            "Region-pixel-count", "Short-line-density-5",
            "Short-line-density-2", "Vedge-mean", "Vegde-sd",
            "Hedge-mean", "Hedge-sd", "Intensity-mean",
            "Rawred-mean", "Rawblue-mean", "Rawgreen-mean",
            "Exred-mean", "Exblue-mean", "Exgreen-mean",
            "Value-mean", "Saturatoin-mean", "Hue-mean", "Class"
        ]
        segment_df = pd.read_csv(os.path.join(self.data_prefix, path),
                        skiprows=24, sep=",", header=None)
        segment_df.columns = columns
        segment_df = segment_df.sample(frac=1.0).reset_index(drop=True)
        segment_df["Class"] = np.where(segment_df["Class"] == " negative", -1, 1)
        return segment_df


class PageBlocks0(Dataset):
    def __init__(self):
        super().__init__("imbalanced", "page-blocks0")

    def _preprocess(self):
        path = "page-blocks0/page-blocks0.dat"
        columns = [
            "Height", "Length", "Area", "Eccen", "P_black",
            "P_and", "Mean_tr", "Blackpix", "Blackand",
            "Wb_trans", "Class"
        ]
        blocks_df = pd.read_csv(os.path.join(self.data_prefix, path),
                        skiprows=15, sep=",", header=None)
        blocks_df.columns = columns
        blocks_df = blocks_df.sample(frac=1.0).reset_index(drop=True)
        blocks_df["Class"] = np.where(blocks_df["Class"] == " negative", -1, 1)
        return blocks_df


class Iris0(Dataset):
    def __init__(self):
        super().__init__("imbalanced", "iris0")

    def _preprocess(self):
        path = "iris0/iris0.dat"
        columns = [
            "SepalLength", "SepalWidth", "PetalLength",
            "PetalWidth", "Class"
        ]
        iris_df = pd.read_csv(os.path.join(self.data_prefix, path),
                    skiprows=9, sep=",", header=None)
        iris_df.columns = columns
        iris_df = iris_df.sample(frac=1.0).reset_index(drop=True)
        iris_df["Class"] = np.where(iris_df["Class"] == " negative", -1, 1)
        return iris_df


class Glass4(Dataset):
    def __init__(self):
        super().__init__("imbalanced", "glass4")

    def _preprocess(self):
        path = "glass4/glass4.dat"
        columns = [
            "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba",
            "Fe", "Class"
        ]
        glass_df = pd.read_csv(os.path.join(self.data_prefix, path),
                    skiprows=14, sep=",", header=None)
        glass_df.columns = columns
        glass_df = glass_df.sample(frac=1.0).reset_index(drop=True)
        glass_df["Class"] = np.where(glass_df["Class"] == " negative", -1, 1)
        return glass_df


class Abalone19(Dataset):
    def __init__(self):
        super().__init__("imbalanced", "abalone19")

    def _preprocess(self):
        path = "abalone19/abalone19.dat"
        columns = [
            "Sex", "Length", "Diameter", "Height", "Whole_weight",
            "Shucked_weight", "Viscera_weight", "Shell_weight",
            "Class"
        ]
        abalone_df = pd.read_csv(os.path.join(self.data_prefix, path),
                    skiprows=13, sep=",", header=None)
        abalone_df.columns = columns
        abalone_df["Sex"] = pd.factorize(abalone_df["Sex"])[0]
        abalone_df = abalone_df.sample(frac=1.0).reset_index(drop=True)
        abalone_df["Class"] = np.where(abalone_df["Class"] == "negative", -1, 1)
        return abalone_df


class Coil2000(Dataset):
    def __init__(self):
        super().__init__("many-attributes", "coil2000")

    def _preprocess(self):
        path = "coil2000/coil2000.dat"
        columns = [
        "MOSTYPE", "MAANTHUI", "MGEMOMV", "MGEMLEEF", "MOSHOOFD",
        "MGODRK", "MGODPR", "MGODOV", "MGODGE", "MRELGE", "MRELSA",
        "MRELOV", "MFALLEEN", "MFGEKIND", "MFWEKIND", "MOPLHOOG",
        "MOPLMIDD", "MOPLLAAG", "MBERHOOG", "MBERZELF", "MBERBOER",
        "MBERMIDD", "MBERARBG", "MBERARBO", "MSKA", "MSKB1", "MSKB2",
        "MSKC", "MSKD", "MHHUUR", "MHKOOP", "MAUT1", "MAUT2", "MAUT0",
        "MZFONDS", "MZPART", "MINKM30", "MINK3045", "MINK4575",
        "MINK7512", "MINK123M", "MINKGEM", "MKOOPKLA", "PWAPART",
        "PWABEDR", "PWALAND", "PPERSAUT", "PBESAUT", "PMOTSCO",
        "PVRAAUT", "PAANHANG", "PTRACTOR", "PWERKT", "PBROM", "PLEVEN",
        "PPERSONG", "PGEZONG", "PWAOREG", "PBRAND", "PZEILPL",
        "PPLEZIER", "PFIETS", "PINBOED", "PBYSTAND", "AWAPART",
        "AWABEDR", "AWALAND", "APERSAUT", "ABESAUT", "AMOTSCO",
        "AVRAAUT", "AAANHANG", "ATRACTOR", "AWERKT", "ABROM", "ALEVEN",
        "APERSONG", "AGEZONG", "AWAOREG", "ABRAND", "AZEILPL",
        "APLEZIER", "AFIETS", "AINBOED", "ABYSTAND", "CLASS"
        ]
        coil_df = pd.read_csv(os.path.join(self.data_prefix, path),
                    skiprows=90, sep=",", header=None)
        coil_df.columns = columns
        coil_df = coil_df.sample(frac=1.0).reset_index(drop=True)
        coil_df["CLASS"] = np.where(coil_df["CLASS"] == 0, -1, 1)
        return coil_df

class Iris(Dataset):
    def __init__(self):
        super().__init__("multiclass", "iris")

    def _preprocess(self):
        path = "iris/iris.dat"
        columns = [
            "SepalLength", "SepalWidth", "PetalLength", "PetalWidth",
            "Class"
        ]
        iris_df = pd.read_csv(os.path.join(self.data_prefix, path),
                              skiprows=9, sep=", ", header=None)
        iris_df.columns = columns
        iris_df = iris_df.sample(frac=1.0).reset_index(drop=True)
        iris_df["Class"] = pd.factorize(iris_df["Class"])[0]
        return iris_df
