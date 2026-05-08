import numpy as np

# MultinomialNaiveBayes class
class MultinomialNaiveBayes():
    def __init__(self):
        self.class_priors = None
        self.feature_log_probs = None

    def fit(self, X, y):
        """
        Fit the Naive Bayes model to the training data X and labels y.
        Returns the class priors and feature log probabilities.
        """
        # Store all unique classes 
        classes = np.unique(y)

        # Compute class priors log(P(y=c) for each class (see theory.md, section 7)
        class_priors = []
        for c in classes:
            class_priors.append(np.log(len(y[y == c]) / len(y)))
            
        # Store log(P(xi|y=c)) for each feature i and class c in a dictionary
        feature_log_probs = []

        # For each class, compute log(P(xi, y=c)) (see theory.md, section 9)
        for c in classes:
            # Extract rows of a sparse matrix corresponding to class c, (shape N_c, V) with Nc lines of class c and V features (vocabulary size)
            class_c = X[y == c]
            # Compute total number of occurences of the word i in all sms of class c
            n_i_c = np.asarray(class_c.sum(axis=0)).flatten() # to return a vector (V,) and not (1, V)
            # Compute P(xi|y=c) 
            p_xi_c = (n_i_c + 1) / (n_i_c.sum() + X.shape[1]) # V=X.shape[1], number of features X
            # Store log(P(xi|y=c)) for each feature i and class c in a dictionary
            feature_log_probs.append(np.log(p_xi_c))

        # Store class_priors and feature_log_probs
        self.class_priors = np.array(class_priors)
        self.feature_log_probs = np.array(feature_log_probs)

    # Prediction pipeline (see theory.md, section 10)
    def predict(self, X):
        """
        Predict the class labels for the input data X using the fitted model.
        Returns the predicted class labels.
        """
        # For each class c, compute the score : score(c) = log(P(y=c)) + sum of i of (xi * log(P(xi, y=c)))
        scores = X @ self.feature_log_probs.T
        scores = scores + self.class_priors

        return np.argmax(scores, axis=1)


if __name__ == "__main__":
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import MultinomialNB as SklearnNB
    import pandas as pd

    # Load dataset
    URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(URL, sep='\t', header=None, names=['label', 'text'])
    df['label'] = df['label'].apply(lambda x: 0 if x == 'ham' else 1)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)

    # Scratch
    clf = MultinomialNaiveBayes()
    clf.fit(X_train_counts, y_train.values)
    y_pred = clf.predict(X_test_counts)
    print(f"Scratch accuracy : {accuracy_score(y_test, y_pred):.4f}")

    # Sklearn
    clf_sk = SklearnNB()
    clf_sk.fit(X_train_counts, y_train)
    y_pred_sk = clf_sk.predict(X_test_counts)
    print(f"Sklearn accuracy : {accuracy_score(y_test, y_pred_sk):.4f}")