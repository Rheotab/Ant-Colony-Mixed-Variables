import numpy as np
import math


class Archive:

    def __init__(self, ndim, k=100, q=0.05, m=5, maximize=True):
        """
            Initializes the archive to store and manage solutions.

            Parameters:
            - ndim: Number of dimensions (variables in the solution).
            - k: Size of the archive (number of stored solutions).
            - q: Influence factor for weighting solutions.
            - m: Number of ants (for Ant Colony Optimization).
            - maximize: Boolean indicating whether the goal is to maximize or minimize the objective function.
        """
        self.k = k
        self.q = q
        self.ndim = ndim
        self.memory = np.zeros((k, ndim + 2))
        self.maximize = maximize
        self.weights_memory = None
        self.categorical_weights = None
        self.sum_probs = None

        self.sig_sampler = {}

        self.mask = np.ones(k + m, dtype=bool)
        self.mask[[i for i in range(k, k + m)]] = False

    def initialize(self, elements):
        """
            Initializes the archive with a set of initial elements (solutions).
        """
        assert isinstance(elements, np.ndarray)
        assert elements.shape[0] == self.memory.shape[0]
        self.memory[:, [i for i in range(elements.shape[1])]] = elements

    def evaluate_all_memory(self, f):
        """
            Evaluates all stored solutions using the given function `f`.
        """
        kk = self.memory[:, [i for i in range(self.ndim)]]
        r = f(kk).reshape(-1)
        self.memory[:, self.ndim] = r

    def sort_memory(self):
        """
            Sorts stored solutions based on their objective function values.
            If maximizing, sorts in descending order; otherwise, ascending.
        """
        sorted_index = np.argsort(self.memory[:, self.ndim])[::-1]
        if not self.maximize:
            sorted_index = sorted_index[::-1]
        self.memory = self.memory[sorted_index]
        pass

    def build_weights(self):
        """
            Builds probability weights for solutions based on ranking.
        """
        if self.weights_memory is None:
            self.weights_memory = [self.weight(i) for i in range(1, self.k + 1)]
            self.sum_probs = np.sum(self.weights_memory)
        self.memory[:, self.ndim + 1] = self.weights_memory

    def weight(self, i):
        """
            Computes the weight for a solution at rank `i` based on a Gaussian-like distribution.
        """
        first_term = 1 / (self.q * self.k * math.sqrt(2 * math.pi))
        second_term = np.exp((-(i - 1) ** 2) / (2 * self.q ** 2 * self.k ** 2))
        w = first_term * second_term
        return w

    def choice(self):
        """
            Selects a solution from the archive probabilistically based on weights.
        """
        indexes = [i for i in range(self.k)]
        probs = self.memory[:, self.ndim + 1] / self.sum_probs
        ind = np.random.choice(a=indexes, size=1, p=probs)
        return self.memory[ind, :].reshape(-1), ind

    def get_sig_to_sample(self, row, idx, xi):
        """
            Computes the standard deviation (sigma) for sampling new solutions.
            This is based on the variation of stored solutions.
        """
        sum = 0
        for iter in range(self.k):
            sum += np.abs(self.memory[iter, idx] - row[idx])
        sig = xi * sum / (self.k - 1)
        return sig

    def build_categorical_weights(self, idx_cat, bounds):
        """
            Builds probability distributions for categorical variables.
        """
        u = {}
        nu = {}
        weights = {}
        for idx in idx_cat:
            u[str(idx)] = {}
            weights[str(idx)] = {}
            domain = bounds[idx]
            column = self.memory[:, idx].astype('int64')
            counter = {}
            for value in domain:
                counter[str(value)] = 0
            for c in column:
                if str(int(c)) in counter:
                    counter[str(int(c))] += 1

            nu[str(idx)] = 0
            for value in domain:
                u[str(idx)][str(value)] = counter[str(value)]
            for e in domain:
                if counter[str(e)] == 0:
                    nu[str(idx)] += 1
            sum = 0
            for value in domain:
                res = 0
                if u[str(idx)][str(value)] > 0:
                    idx_best = np.where(column == value)[0][0]
                    w = self.memory[idx_best, self.ndim + 1]
                    res += w / u[str(idx)][str(value)]
                if nu[str(idx)] > 0:
                    res += self.q / nu[str(idx)]
                weights[str(idx)][str(value)] = res
                sum += res
            for key in weights[str(idx)]:
                weights[str(idx)][key] = weights[str(idx)][key] / sum
        self.categorical_weights = weights

    def get_probs(self, idx):
        """
            Retrieves probability distribution for categorical variables.
        """
        values = list(self.categorical_weights[str(idx)].keys())
        weights = [self.categorical_weights[str(idx)][val] for val in values]
        values = list(np.int_(values))
        return values, weights

    def store_evaluate_results(self, ants, f=None):
        """
            Stores solutions generated by ants and evaluates their fitness.
        """
        solutions = np.zeros((len(ants), self.ndim + 2))
        solutions[:, [i for i in range(self.ndim)]] = np.asarray([ant.solution.copy() for ant in ants])
        if f is not None:
            solutions[:, self.ndim] = f(solutions[:, [i for i in range(self.ndim)]]).reshape(-1)
        self.memory = np.append(self.memory, solutions, axis=0)  # Append rows

    def select(self):
        """
            Selects the top solutions from the archive based on the mask.
        """
        self.memory = self.memory[self.mask, :]

    def get_best_indiv(self):
        """
        Returns the best solution and its objective function value.
        """
        return self.memory[0, [i for i in range(0, self.ndim)]], self.memory[0, self.ndim]

    def reset(self):
        """
            Resets the archive by clearing stored solutions.
        """
        self.memory = np.zeros((self.k, self.ndim + 2))  # + 2 is for Fitness and weight.

    def reset_sig_sampler(self):
        """
            Resets the sigma sampling dictionary.
        """
        self.sig_sampler = {}
