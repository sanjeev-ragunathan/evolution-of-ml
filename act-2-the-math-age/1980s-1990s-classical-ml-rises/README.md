# Classical ML Rises (1980s–1990s)

## Neural nets were back... but were they needed?

Backpropagation worked. Neural nets could solve XOR and more. But they were slow, hard to train, and hungry for data that didn't exist yet. Meanwhile, mathematicians and statisticians looked at the same problems and asked: **why complicate things?**

They didn't need layers and backpropagation. They had elegant math. And for the problems people cared about in the 80s and 90s - predicting prices, classifying documents, diagnosing patients - their tools were faster, simpler, and often better.

This is how a whole different branch of ML emerged. Not neurons and biology - **pure mathematics and statistics**.

## The toolbox grows

These models didn't replace each other — they each brought a different strength. The toolbox grew over time:

**k-Nearest Neighbors (1951)** — The simplest idea in all of ML. Want to classify something? Find the k most similar examples in your data and let them vote. No training, no weights. Just "show me something similar." Simple, but limited — slow on large datasets and sensitive to irrelevant features.

**Decision Trees (1986)** — Ask yes/no questions about the data. "Is the passenger female? Is she older than 30? First class?" Split the data at each question until each group is pure. Intuitive and interpretable — you can read the tree and understand *why* it made a decision. But a single tree overfits easily — it memorizes the training data instead of learning general patterns.

**Support Vector Machines (1995)** — Remember how the perceptron finds *any* line that separates data? SVMs find the **best** line — the one with the widest margin between classes. The "kernel trick" lets them draw curved boundaries, not just straight lines. Powerful, but sensitive to scale (more on that later).

**Random Forests (2001)** — The fix for overfitting trees. Grow 100 trees, each on a random slice of the data, and let them vote. Individual trees make mistakes, but their errors cancel out. The wisdom of the crowd, applied to algorithms.

## Why libraries? Why now?

Remember in Act 1, you wrote nested loops for every weighted sum:

```python
for i in range(hidden_size):
    for j in range(input_size):
        weighted_sum += weights[i][j] * inputs[j]
```

With 1000 inputs and 500 neurons, that's **500,000 multiplications** — in slow Python loops. Now multiply that by thousands of epochs and millions of data points. Days of training.

**NumPy** does the same math in optimized C code. What takes Python loops 10 seconds takes NumPy a fraction of a millisecond:

```python
output = np.dot(weights, inputs)  # same 500,000 multiplications, one line
```

Same math. Same result. Orders of magnitude faster. You earned these libraries because the scale demands them:

- **pandas** - load and manipulate tabular data (Python's spreadsheet)
- **NumPy** - fast math on arrays (runs in C under the hood)
- **Scikit-learn** - every classical ML algorithm in one consistent interface
- **matplotlib** - plotting and visualization

But because you built everything from scratch in Act 1, none of this is magic. When Scikit-learn trains a model, you *know* it's doing the same loops you wrote - just faster.

## The full pipeline

This era established the standard ML workflow:

**1. Load data** — `pd.read_csv("titanic.csv")` - real passengers from the real Titanic

**2. Feature engineering** — which columns matter?
```
Pclass    1 → 63% survived    3 → 24% survived
Sex       female → 74%        male → 19%
```
"Women and children first" wasn't just a saying. First class passengers were closer to the lifeboats. The data tells a story about 1912 society.

**3. Clean messy data** - real-world data has gaps
- 177 missing ages → fill with the mean (imputation)
- 687 missing cabins → drop the column (77% empty)
- Text columns like "male"/"female" → convert to numbers (encoding)

**4. Split features** - separate what the model sees (X) from what it predicts (Y). Then split into **training data** and **test data** - because testing on the data you trained on is cheating. The model might just memorize answers instead of learning patterns.
```python
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```
`test_size=0.2` basically means, test - 20% and train - 80%.

**5. Build model** - Scikit-learn's pattern for every algorithm:
```python
model = SomeAlgorithm()      # create
model.fit(X_train, Y_train)  # train
model.predict(X_test)         # predict
```

Same interface you built — `train` and `predict`. Scikit-learn just standardized it.

## Why Decision Tree and Random Forest were fine but KNN and SVM were not

First results:
```
Decision Tree:   0.7765
k-NN:            0.6927
SVM:             0.6536
Random Forest:   0.8156
```

k-NN and SVM did worse than a simple tree. The culprit: **scale**.

`Fare` ranges from 0–512. `Sex` is 0 or 1. When k-NN measures distance between passengers, a difference of 500 in fare **drowns out** a difference of 1 in sex. The model thinks fare matters 500x more than gender - just because the numbers are bigger.

Decision trees don't care about scale - they just split on "is fare above or below 50?" The actual magnitude doesn't matter. That's why the tree was fine.

k-NN and SVM are **distance-based models** - sensitive to scale.

**Fix: feature scaling** - squish all features to the same range using `StandardScaler`.

After scaling:
```
Decision Tree:   0.7765  (unchanged — trees ignore scale)
k-NN:            0.8045  (0.69 → 0.80)
SVM:             0.8212  (0.65 → 0.82)
Random Forest:   0.8156
```

SVM went from dead last to first place. Same algorithm, same data. Just proper scaling.

## Key Insight

This is the core lesson of the Math Age: **the algorithm matters less than how you prepare the data.** Feature engineering, cleaning, encoding, scaling — this is where the real skill lived. A good data scientist with a simple model beat a lazy one with a fancy model every time.

---

> **What's `random_state=42`?** - These algorithms involve randomness (shuffling data, sampling features). Setting `random_state` to any number makes the randomness reproducible - same result every run. 42 is just a convention from *The Hitchhiker's Guide to the Galaxy*: "the answer to the ultimate question of life, the universe, and everything." The ML community adopted it as the default seed. The number itself is meaningless.