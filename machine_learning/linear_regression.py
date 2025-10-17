"""
Linear regression is the most basic type of regression commonly used for
predictive analysis. The idea is pretty simple: we have a dataset and we have
features associated with it. Features should be chosen very cautiously
as they determine how much our model will be able to make future predictions.
We try to set the weight of these features, over many iterations, so that they best
fit our dataset. In this particular code, I had used a CSGO dataset (ADR vs
Rating). We try to best fit a line through dataset and estimate the parameters.
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
#     "numpy",
#     "matplotlib",
# ]
# ///

import httpx
import numpy as np
import matplotlib.pyplot as plt


def collect_dataset():
    """Collect dataset of CSGO (ADR vs Rating)."""
    response = httpx.get(
        "https://raw.githubusercontent.com/yashLadha/The_Math_of_Intelligence/"
        "master/Week1/ADRvsRating.csv",
        timeout=10,
    )
    lines = response.text.splitlines()
    data = []
    for item in lines:
        item = item.split(",")
        data.append(item)
    data.pop(0)  # remove header
    dataset = np.array(data, dtype=float)
    return dataset


def run_steep_gradient_descent(data_x, data_y, len_data, alpha, theta):
    """Perform one step of gradient descent."""
    n = len_data
    prod = np.dot(theta, data_x.T)
    prod -= data_y.T
    sum_grad = np.dot(prod, data_x)
    theta = theta - (alpha / n) * sum_grad
    return theta


def sum_of_square_error(data_x, data_y, len_data, theta):
    """Compute mean squared error."""
    prod = np.dot(theta, data_x.T)
    prod -= data_y.T
    sum_elem = np.sum(np.square(prod))
    error = sum_elem / (2 * len_data)
    return error


def run_linear_regression(data_x, data_y, iterations=20000, alpha=0.000155):
    """Train a linear regression model using gradient descent."""
    no_features = data_x.shape[1]
    len_data = data_x.shape[0]
    theta = np.zeros((1, no_features))

    for i in range(iterations):
        theta = run_steep_gradient_descent(data_x, data_y, len_data, alpha, theta)
        if i % 2000 == 0:  # print error every 2000 iterations
            error = sum_of_square_error(data_x, data_y, len_data, theta)
            print(f"Iteration {i + 1} - Error: {error:.5f}")
    return theta


def mean_absolute_error(predicted_y, original_y):
    """Compute Mean Absolute Error (MAE)."""
    total = sum(abs(y - predicted_y[i]) for i, y in enumerate(original_y))
    return total / len(original_y)

#  New function added -plot :

def plot_regression_line(data_x, data_y, theta):
    """Plot the dataset and the fitted regression line."""
    adr = data_x[:, 1]  # second column is ADR (feature)
    rating = data_y     # actual Rating values

    # predicted Rating values from the model
    predicted_rating = theta[0, 0] + theta[0, 1] * adr

    plt.figure(figsize=(8, 6))
    plt.scatter(adr, rating, color="blue", label="Actual Data")
    plt.plot(adr, predicted_rating, color="red", linewidth=2, label="Best Fit Line")
    plt.title("Linear Regression: ADR vs Player Rating")
    plt.xlabel("ADR (Average Damage per Round)")
    plt.ylabel("Player Rating")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def main():
    """Main driver function."""
    data = collect_dataset()

    len_data = data.shape[0]
    data_x = np.c_[np.ones(len_data), data[:, :-1]]  # add intercept
    data_y = data[:, -1]

    theta = run_linear_regression(data_x, data_y)

    print("\nResultant Feature Vector (θ):")
    for i in range(theta.shape[1]):
        print(f"θ[{i}] = {theta[0, i]:.5f}")

    # Plot regression line
    plot_regression_line(data_x, data_y, theta)


if __name__ == "__main__":
    main()

