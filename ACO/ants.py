import numpy as np
import scipy.stats


class Ant:
    """
        Represents an "ant" in an Ant Colony Optimization (ACO) algorithm.
        The ant builds a solution using pheromone-influenced sampling from an archive.
    """

    def __init__(self, ndim, idx_cont, idx_ord, idx_cat, archive, xi, bounds):
        """
            Initializes an ant with solution space dimensions and relevant indices.

            Parameters:
            - ndim: Number of dimensions in the solution space.
            - idx_cont: Indices of continuous variables.
            - idx_ord: Indices of ordinal variables.
            - idx_cat: Indices of categorical variables.
            - archive: ACO archive storing previous solutions and pheromone data.
            - xi: Parameter influencing pheromone-based sampling.
            - bounds: Bounds for continuous variables.
        """
        self.ndim = ndim
        self.idx_cont = idx_cont
        self.idx_ord = idx_ord
        self.idx_cat = idx_cat
        self.archive = archive
        self.xi = xi
        self.bounds = bounds

        self.solution = np.zeros(ndim)
        for cont in idx_cont:
            self.solution[cont] = np.random.uniform(low=bounds[cont][0], high=bounds[cont][1])

    def build_solution(self):
        """
            Constructs a complete solution by sampling from the ACO archive.
            - Continuous variables are sampled using pheromone-based Gaussian sampling.
            - Ordinal variables are treated as continuous and then rounded.
            - Categorical variables are sampled using probability distributions from the archive.
        """
        self.solution[self.idx_cont] = self.aco_r(self.idx_cont)
        self.solution[self.idx_ord] = self.acomv_o()
        self.solution[self.idx_cat] = self.acomv_c(self.idx_cat)

    def aco_r(self, index):
        """
            Samples new values for continuous variables using pheromone-based normal distribution.

            Parameters:
            - index: Indices of the continuous variables.

            Returns:
            - Updated values for the given indices.
        """
        res = self.solution.copy()
        row, _ind = self.archive.choice()
        for idx in index:
            mu = row[idx]
            sig = self.archive.get_sig_to_sample(row, idx, self.xi)
            if sig == 0:
                g = mu
            else:
                g = self.sample_g(mu, sig, idx)
            res[idx] = g
        return res[index]

    def sample_g(self, mu, sig, index):
        """
           Samples a value from a truncated normal distribution.

           Parameters:
           - mu: Mean of the distribution.
           - sig: Standard deviation.
           - index: Index of the variable in the solution.

           Returns:
           - Sampled value within bounds.
       """
        lower = self.bounds[index][0]
        upper = self.bounds[index][1]
        res = scipy.stats.truncnorm.rvs((lower - mu) / sig, (upper - mu) / sig, loc=mu, scale=sig).item()
        return res

    def acomv_o(self):
        """
           Samples values for ordinal variables.
           - Uses continuous relaxation and then rounds to the nearest integer.

           Returns:
           - Rounded values for ordinal variables.
       """
        continuous_relaxation = self.aco_r(self.idx_ord)
        res = np.rint(continuous_relaxation)
        return res

    def acomv_c(self, index):
        """
           Samples values for categorical variables based on archive probabilities.

           Parameters:
           - index: Indices of categorical variables.

           Returns:
           - Sampled categorical values.
       """
        res = self.solution.copy()
        for idx in index:
            val, probs = self.archive.get_probs(idx)
            if len(val) == 1:
                res[idx] = val[0]
            else:
                res[idx] = np.random.choice(val, size=1, p=probs)
        return res[index]
