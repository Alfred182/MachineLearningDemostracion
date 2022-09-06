from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt


def get_data():
    data = load_iris()
    x = data.data
    y = data.target
    return x, y


def plot(model):
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        plt.plot(X_plot[:, 0], X_plot[:, 1], linestyle='none', marker='o', label=target_name)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.axis('equal')
    plt.legend();
    plt.subplot(1, 2, 2)
    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        plt.plot(X_plot[:, 2], X_plot[:, 3], linestyle='none', marker='o', label=target_name)
    plt.xlabel(feature_names[2])
    plt.ylabel(feature_names[3])
    plt.axis('equal')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(model,
                       feature_names=iris.feature_names,
                       class_names=iris.target_names,
                       filled=True)
    fig.savefig("GeneratedDecistionTree.png")


x, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=50, test_size=0.25)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Train data accuracy:", accuracy_score(y_true=y_train, y_pred=model.predict(X_train)))
print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred=y_pred))

text_representation = tree.export_text(model)
print(text_representation)
plot(model)
