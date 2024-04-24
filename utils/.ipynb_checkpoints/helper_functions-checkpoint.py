import numpy as np
from scipy.stats import norm
import warnings
import jax.numpy as jnp
import jax

# Assuming params_dict is a dictionary with keys of type str and values of type jnp.ndarray
def save_params_dict(params_dict, filename):
    with open(filename, 'wb') as f:
        jnp.save(filename, params_dict)
    f.close()
# @title Utility functions

def find_nearest_neighbors(reference_idx, embedding, num_neighbors=9):
    """Find the indices of the nearest neighbors to a reference point."""
    reference_point = embedding[reference_idx]
    distances = distance.cdist([reference_point], embedding, 'euclidean').flatten()
    nearest_indices = np.argsort(distances)[1:num_neighbors+1]  # Exclude the reference point itself
    warnings.filterwarnings("ignore", category=UserWarning)  # Ignore userwarnings
    return nearest_indices

def find_indices(original, search):
    indices = []
    for row in search:
        # Find the index of the row in the original array
        index = np.where((original == row).all(axis=1))[0]
        if index.size > 0:
            indices.append(index[0])
        else:
            indices.append(-1)  # -1 indicates not found
    return np.array(indices)

def are_points_separated(point, other_points, min_distance):
    """Check if 'point' is at least 'min_distance' away from all points in 'other_points'."""
    return np.all(np.linalg.norm(other_points - point, axis=1) >= min_distance)


def load_params(filename):
    with open(filename,'rb') as f:
        val = jnp.load(filename,allow_pickle=True)
    f.close()
    return val.item()


def probability_of_improvement(y_mean, y_std, best_observed_value:float=0.01):
    z = (y_mean - best_observed_value) / y_std
    pi = norm.cdf(z)
    return pi

def expected_improvement(y_mean, y_std, best_observed_value:float=0.01):
    z = (y_mean - best_observed_value) / y_std
    ei = (y_mean - best_observed_value) * norm.cdf(z) + y_std * norm.pdf(z)
    return ei

def lower_confidence_bound(y_mean, y_std, beta_observed_value:float=1.0):
    lcb = y_mean - beta_observed_value * y_std
    return lcb

def entropy(y_mean, y_std, best_observed_value:float=0.01):
    entropy = -norm.pdf((y_mean - best_observed_value) / y_std) * (y_mean - best_observed_value + y_std * norm.pdf((y_mean - best_observed_value) / y_std))
    return entropy

def upper_confidence_bound(y_mean, y_std, beta_observed_value:float=10.0):
    ucb = y_mean + beta_observed_value * y_std
    return ucb