# Wine-Quality-Feature-Selection
Implementation of forward selection and backward elimination from scratch using Naive Bayes for feature selection.

## Forward Selection

Forward selection is a feature selection technique that incrementally builds a model by adding one feature at a time, starting with an empty set of features. At each step, it evaluates the performance of the model with the added feature and selects the feature that provides the highest improvement in a specified evaluation metric (e.g., accuracy or loss). This process continues until the desired number of features is reached or no further improvement can be achieved.

To perform forward selection using this tool, use the `forward_selection` method and specify the number of features to select and the evaluation metric.

## Backward Elimination

Backward elimination, on the other hand, starts with all available features and iteratively removes the least significant feature at each step. Similar to forward selection, it evaluates the model's performance using a specified metric and removes the feature that results in the least degradation of the model. This process continues until the desired number of features is reached or further removal negatively impacts the model's performance.

To perform backward elimination using this tool, use the `backward_elimination` method and specify the number of features to select and the evaluation metric.

## Usage

```python
import pandas as pd
from feature_selection import FeatureSelection
from sklearn.model_selection import train_test_split

df = pd.read_csv("wine_quality.csv")
df.dropna(inplace=True)

x = df.drop(["type"], axis=1)
y = df["type"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

FS = FeatureSelection(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
FS.backward_elimination(metric="loss", feature_number=11)
# FS.forward_selection(metric="loss", feature_number=4)

print(FS.get_best_features())


