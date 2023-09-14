import warnings
from typing import List

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class FeatureSelection:
    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        self.x_train = x_train
        self.x_test = x_test
        encoder = LabelEncoder()
        self.y_train = encoder.fit_transform(y_train)
        self.y_test = encoder.fit_transform(y_test)

    def forward_selection(
        self, feature_number: int = -1, metric: str = "accuracy"
    ) -> List[float]:
        if feature_number == -1:
            feature_number = len(list(self.x_train.columns))
        self.selected_features_along_with_acc = {}

        features = list(self.x_train.columns)
        comb = []
        for _ in range(len(features)):
            comb_acc = {}

            for feature in features:
                if feature in comb:
                    continue

                NB = GaussianNB()
                NB.fit(self.x_train[comb + [feature]], self.y_train)

                y_pred = NB.predict(self.x_test[comb + [feature]])
                if metric == "accuracy":
                    test_acc = accuracy_score(y_true=self.y_test, y_pred=y_pred)
                if metric == "loss":
                    test_acc = log_loss(
                        y_true=self.y_test, y_pred=y_pred, labels=["white", "red"]
                    )

                comb_acc[tuple(comb + [feature])] = test_acc

            best_comb = max(comb_acc, key=comb_acc.get)
            self.selected_features_along_with_acc[best_comb] = comb_acc[best_comb]
            comb = list(best_comb)

            if len(list(best_comb)) == feature_number:
                break

        return [
            list(self.selected_features_along_with_acc.keys())[-1],
            list(self.selected_features_along_with_acc.values())[-1],
        ]

    def backward_elimination(
        self, feature_number: int = -1, metric: str = "accuracy"
    ) -> List[float]:
        if feature_number == -1:
            feature_number = len(list(self.x_train.columns))
        self.selected_features_along_with_acc = {}

        features = list(self.x_train.columns)
        comb = features
        for _ in range(len(features) - 1, 0, -1):
            comb_acc = {}

            for feature in features:
                if feature not in comb:
                    continue

                comb.remove(feature)
                NB = GaussianNB()
                NB.fit(self.x_train[comb], self.y_train)

                y_pred = NB.predict(self.x_test[comb])
                if metric == "accuracy":
                    test_acc = accuracy_score(y_true=self.y_test, y_pred=y_pred)
                if metric == "loss":
                    test_acc = log_loss(
                        y_true=self.y_test, y_pred=y_pred, labels=["white", "red"]
                    )
                comb_acc[tuple(comb)] = test_acc

                comb.append(feature)

            best_comb = max(comb_acc, key=comb_acc.get)
            self.selected_features_along_with_acc[best_comb] = comb_acc[best_comb]
            comb = list(best_comb)

            if len(list(best_comb)) == feature_number:
                break

        return [
            list(self.selected_features_along_with_acc.keys())[-1],
            list(self.selected_features_along_with_acc.values())[-1],
        ]

    def get_best_features(self):
        best_features = max(
            self.selected_features_along_with_acc,
            key=self.selected_features_along_with_acc.get,
        )
        acc = self.selected_features_along_with_acc[best_features]

        return best_features, acc
