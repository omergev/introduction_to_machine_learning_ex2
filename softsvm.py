import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt

# todo: complete the following functions, you may add auxiliary functions or define class to help you


def softsvm(l, train_x: np.array, train_y: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param train_x: numpy array of size (m, d) containing the training sample
    :param train_y: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m, d = train_x.shape  # m: number of samples, d: number of features

    # Construct matrix H as blocking matrix in size of (d + m) X (d + m)
    H = spmatrix(2 * l * 1.0, range(d), range(d), (d + m, d + m))

    # Construct vector u = [0...0, 1/m...1/m] zeros are first d enters and 1/m are m next enters (size of d + m)
    u = np.append(np.zeros(d), np.ones(m) / m)
    u = matrix(u)

    # Construct matrix A as blocking matrix in size of 2m X (d + m)
    zeros_matrix = spmatrix(0.0, [], [], (m, d))
    identity_matrix = spmatrix(1.0, range(m), range(m))

    # Compute diag(train_y) * train_x using sparse matrix multiplication
    diag_y = spdiag(matrix(train_y))
    product_matrix = diag_y * matrix(train_x)

    # Construct A using sparse block matrices
    col1_top = zeros_matrix  # m X d
    col1_bottom = product_matrix  # m X d

    col2_top = identity_matrix  # m X m
    col2_bottom = identity_matrix  # m X m
    A = sparse([[col1_top, col1_bottom], [col2_top, col2_bottom]])

    # Construct vector v
    v = matrix(np.concatenate((np.zeros(m), np.ones(m))))
    v = matrix(v)
    # Solve the quadratic programming problem
    sol = solvers.qp(H, u, -A, -v)

    # Extract w
    w = np.array(sol["x"][:d])

    return w.reshape(-1, 1)


def simple_test():
    # load question 2 data
    data = np.load("EX2q2_mnist.npz")
    train_x = data["Xtrain"]
    test_x = data["Xtest"]
    train_y = data["Ytrain"]
    test_y = data["Ytest"]

    m = 100
    d = train_x.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(train_x.shape[0])
    _train_x = train_x[indices[:m]]
    _train_y = train_y[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _train_x, _train_y)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(
        w, np.ndarray
    ), "The output of the function softsvm should be a numpy array"
    assert (
        w.shape[0] == d and w.shape[1] == 1
    ), f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, test_x.shape[0])
    predicty = np.sign(test_x[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


# Question 2:
def load_data():
    # Load data from 'EX2q2_mnist.npz'
    data = np.load("EX2q2_mnist.npz")
    return data["Xtrain"], data["Ytrain"], data["Xtest"], data["Ytest"]


def compute_error(y_true, x, w):
    y_pred = np.sign(x @ w).flatten()
    return np.mean(y_true != y_pred)  # y<w,x>


def run_experiment(train_x, train_y, test_x, test_y, lambdas, sample_size, num_runs=1):
    training_errors = []
    test_errors = []

    for l in lambdas:
        train_errors_for_lambda = []
        test_errors_for_lambda = []

        for _ in range(num_runs):
            random_indices = np.random.permutation(len(train_x))[:sample_size]
            # crate simples from the train matrices
            sample_x, sample_y = train_x[random_indices], train_y[random_indices]

            w = softsvm(l, sample_x, sample_y)

            # Compute errors
            train_error = compute_error(sample_y, sample_x, w)
            test_error = compute_error(test_y, test_x, w)

            train_errors_for_lambda.append(train_error)
            test_errors_for_lambda.append(test_error)

        training_errors.append(
            (
                np.mean(train_errors_for_lambda),
                np.max(train_errors_for_lambda),
                np.min(train_errors_for_lambda),
            )
        )
        test_errors.append(
            (
                np.mean(test_errors_for_lambda),
                np.max(test_errors_for_lambda),
                np.min(test_errors_for_lambda),
            )
        )

    return training_errors, test_errors


def plot_results(experiments_data: dict):
    # Extract average, max, and min errors for training and test
    avg_train_errors, max_train_errors, min_train_errors = zip(
        *experiments_data["experiment1"]["train_errors"]
    )
    avg_test_errors, max_test_errors, min_test_errors = zip(
        *experiments_data["experiment1"]["test_errors"]
    )

    # Plotting
    plt.figure(figsize=(10, 6))

    # Scatter plot for max, min, and average training errors
    plt.scatter(
        experiments_data["experiment1"]["lambdas"],
        avg_train_errors,
        color="blue",
        label="Avg Train Error [exp 1]",
    )
    plt.scatter(
        experiments_data["experiment1"]["lambdas"],
        avg_test_errors,
        color="black",
        label="Avg Test Error [exp 1]",
    )

    # Calculate 'up' and 'down' for error bars (train)
    up_train = [m - a for m, a in zip(max_train_errors, avg_train_errors)]
    down_train = [a - m for a, m in zip(avg_train_errors, min_train_errors)]
    y_err_train = [down_train, up_train]

    # Calculate 'up' and 'down' for error bars (test)
    up_test = [m - a for m, a in zip(max_test_errors, avg_test_errors)]
    down_test = [a - m for a, m in zip(avg_test_errors, min_test_errors)]
    y_err_test = [down_test, up_test]

    # Plot error bars for training and testing
    plt.errorbar(
        experiments_data["experiment1"]["lambdas"],
        avg_train_errors,
        yerr=y_err_train,
        elinewidth=2,
        capsize=5,
        fmt="r--o",
        color="blue",
        ecolor="lightblue",
        label="Train Error [exp 1]",
    )
    plt.errorbar(
        experiments_data["experiment1"]["lambdas"],
        avg_test_errors,
        yerr=y_err_test,
        elinewidth=2,
        capsize=5,
        fmt="r--o",
        color="black",
        ecolor="gray",
        label="Test Error [exp 1]",
    )

    # Experiment 2:
    avg_train_errors_exp2, max_train_errors, min_train_errors = zip(
        *experiments_data["experiment2"]["train_errors"]
    )
    avg_test_errors_exp2, max_test_errors, min_test_errors = zip(
        *experiments_data["experiment2"]["test_errors"]
    )

    plt.scatter(
        experiments_data["experiment2"]["lambdas"],
        avg_train_errors_exp2,
        color="green",
        label="Avg Train Error [exp 2]",
        zorder=5,
    )
    plt.scatter(
        experiments_data["experiment2"]["lambdas"],
        avg_test_errors_exp2,
        color="orange",
        label="Avg Test Error [exp 2]",
        zorder=5,
    )

    plt.xlabel("λ (log scale)")
    plt.ylabel("Error")
    plt.xscale("log")
    plt.title("Training and Test Errors vs λ")
    plt.legend()
    plt.show()


def experiment1(train_x, train_y, test_x, test_y):
    lambdas_exp1 = [10**n for n in range(1, 11)]
    train_errors_exp1, test_errors_exp1 = run_experiment(
        train_x, train_y, test_x, test_y, lambdas_exp1, sample_size=100, num_runs=10
    )
    return lambdas_exp1, train_errors_exp1, test_errors_exp1


def experiment2(train_x, train_y, test_x, test_y):
    lambdas_exp2 = [10**n for n in [1, 3, 5, 8]]
    train_errors_exp2, test_errors_exp2 = run_experiment(
        train_x, train_y, test_x, test_y, lambdas_exp2, sample_size=1000
    )
    return lambdas_exp2, train_errors_exp2, test_errors_exp2


if __name__ == "__main__":
    # before submitting, make sure that the function simple_test runs without errors
    train_x, train_y, test_x, test_y = load_data()
    lambdas_exp1, train_errors_exp1, test_errors_exp1 = experiment1(
        train_x, train_y, test_x, test_y
    )
    lambdas_exp2, train_errors_exp2, test_errors_exp2 = experiment2(
        train_x, train_y, test_x, test_y
    )
    experiments_data = {
        "experiment1": {
            "lambdas": lambdas_exp1,
            "train_errors": train_errors_exp1,
            "test_errors": test_errors_exp1,
        },
        "experiment2": {
            "lambdas": lambdas_exp2,
            "train_errors": train_errors_exp2,
            "test_errors": test_errors_exp2,
        },
    }
    plot_results(experiments_data=experiments_data)
    # simple_test()

    # here you may add any code that uses the above functions to solve question 2
