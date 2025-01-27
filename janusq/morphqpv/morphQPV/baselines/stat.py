import numpy as np
import scipy.stats as stats
def get_nimimal_shots(probs, eps=1e-2):
    """Get the minimal shots to estimate the probabilities."""
    return int(np.sum(np.ceil(1/eps**2*np.log(1/(1-probs)))))*10


def chi_square_test(probs, truth_probs, alpha=0.0001):
    """Chi square test."""
    critical_statistic = stats.chi2.ppf(1-alpha, len(probs)-1)
    chi_square,p = stats.chisquare(probs,truth_probs,ddof=0)
    return chi_square < critical_statistic

def assertion(target_state,alpha=0.01):
    """Assert that the projection matrix is correct."""
    probs = np.abs(target_state)**2
    shots = get_nimimal_shots(probs, eps=alpha)
    return shots
if __name__ == "__main__":
    ## 8 qubits
    for i in range(4,10):
        random_state = np.random.rand(2**i)
        random_distribution = np.abs(random_state)**2/np.sum(np.abs(random_state)**2)
        print(get_nimimal_shots(np.array(random_distribution)))

    