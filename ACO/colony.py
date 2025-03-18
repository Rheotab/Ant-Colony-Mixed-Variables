import numpy as np

import matplotlib.pyplot as plt

from ACO.ants import Ant
from ACO.archive import Archive


class Colony:
    """
        Represents a colony of ants for an Ant Colony Optimization (ACO) algorithm.
        The colony consists of multiple ants that explore the solution space using pheromone-based learning.
        """

    def __init__(self, bounds, decision_variables, ndim, idx_ord=None, idx_cat=None, criterion=2000, maximize=False,
                 debug=False):
        """
        Initializes the colony with optimization parameters.

        Parameters:
        - bounds: The boundaries for the decision variables.
        - decision_variables: Dictionary containing ACO parameters (m, q, k, xi, etc.).
        - ndim: Number of dimensions in the solution space.
        - idx_ord: Indices of ordinal variables.
        - idx_cat: Indices of categorical variables.
        - criterion: Number of iterations before stopping.
        - maximize: Whether to maximize or minimize the objective function.
        - debug: Enables debugging features such as category tracking.
        """
        self.bounds = bounds
        self.ndim = ndim
        self.maximize = maximize

        # Store indexes, it can be usefull (ordinals may be hard to introduced in GPyOpt)
        self.idx_cat = idx_cat
        self.idx_ord = idx_ord
        self.idx_cont = []
        for i in range(ndim):
            if i not in idx_ord and i not in idx_cat:
                self.idx_cont.append(i)

        # decision variables
        self.m = decision_variables['m']  # m ants
        self.q = decision_variables['q']  # q influence of best sol in archive
        self.k = decision_variables['k']  # Archive sizze
        self.xi = decision_variables['xi']  # width of the search
        self.MaxStagIter = decision_variables['MaxStagIter']  # Stagnating restart
        self.eps = decision_variables['eps']  # Threshold, stoping criteria

        self.archive = Archive(ndim, self.k, m=self.m, maximize=self.maximize)
        self.ants = self.create_ants()

        self.f = None
        self.x0 = None

        self.criterion = criterion
        self.x_best_res = []

        self.best_res = []

        self.debug = debug
        if self.debug:
            self.explored_categories = []
            ii = 0
            for idx_cat in self.idx_cat:
                self.explored_categories.append({})
                for values in self.bounds[idx_cat]:
                    self.explored_categories[ii][str(values)] = [0]
                ii += 1
            self.weights_history = []

    def create_ants(self):
        """
          Creates a list of ants based on the colony parameters.
          Each ant is initialized with the problem's structure.
        """
        return [Ant(self.ndim,
                    self.idx_cont,
                    self.idx_ord,
                    self.idx_cat,
                    self.archive,
                    self.xi,
                    self.bounds) for nb_ant in range(self.m)]

    def optimize(self, x0, f):
        """
               Runs the ACO optimization process.

               Parameters:
               - x0: Initial solution.
               - f: Objective function to be optimized.

               Returns:
               - res_x: Best solution found.
               - res_f: Best function value found.
       """
        self.f = f
        self.x0 = x0
        self.initialize()
        criterion = self.criterion
        x1, fx1 = x0, self.f(x0)
        f_start = fx1
        restart_counter = 0
        x_best, f_best = x1.copy(), fx1
        while criterion > 0:
            if restart_counter > self.MaxStagIter:
                restart_counter = 0
                self.restart()

            self.archive.reset_sig_sampler()
            for ant in self.ants:
                ant.build_solution()
            self.archive.store_evaluate_results(self.ants, f=self.f)
            self.archive.sort_memory()
            self.archive.select()
            self.archive.build_weights()
            self.archive.build_categorical_weights(self.idx_cat, bounds=self.bounds)

            if self.debug:
                # Count how many times every category is selected
                ii = 0
                for idx_cat in self.idx_cat:
                    counter_categ = {}
                    for possible_values in self.bounds[idx_cat]:
                        counter_categ[str(possible_values)] = self.explored_categories[ii][str(possible_values)][
                            len(self.explored_categories[ii][str(possible_values)]) - 1]

                    for ant in self.ants:
                        val = int(ant.solution[idx_cat])
                        counter_categ[str(val)] += 1
                    for possible_values in self.bounds[idx_cat]:
                        self.explored_categories[ii][str(possible_values)].append(counter_categ[str(possible_values)])
                    ii += 1

                # Get categorical wieghts
                self.weights_history.append(self.archive.categorical_weights)

            # Restart / Improvement
            x, fx = self.archive.get_best_indiv()
            if self.maximize:
                if fx > f_best:
                    f_best = fx
                    x_best = x.copy()
            else:
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()

            self.best_res.append(f_best)
            self.x_best_res.append(x_best)

            if self.maximize:
                improvement = fx - fx1
            else:
                improvement = fx1 - fx
            if improvement == 0 or improvement < 0:
                restart_counter += 1
            else:
                restart_counter = 0

            x1, fx1 = x.copy(), fx

            criterion -= 1

        if self.maximize:
            improvement_from_xo = f_best - f_start
        else:
            improvement_from_xo = f_start - f_best
        if improvement_from_xo > 0:
            pass
        if self.maximize:
            arg_max = np.argmax(self.best_res)
        else:
            arg_max = np.argmin(self.best_res)
        res_x = self.x_best_res[arg_max]
        res_f = self.best_res[arg_max]

        if self.debug:
            ii = 0
            for idx_cat in self.idx_cat:
                for val in self.explored_categories[ii]:
                    plt.plot(self.explored_categories[ii][val], label=val)
                plt.legend()
                plt.title(idx_cat)
                plt.show()
                for val in self.bounds[idx_cat]:
                    hist = [self.weights_history[kk][str(idx_cat)][str(val)] for kk in range(len(self.weights_history))]
                    plt.plot(hist, label=val)
                plt.legend()
                plt.title("weights : " + str(idx_cat))
                plt.show()
                ii += 1

        return res_x, res_f

    def restart(self):
        """
            Resets the archive and reinitializes the ants.
        """
        self.archive.reset()
        self.ants = self.create_ants()
        self.initialize()

    def initialize(self):
        """
            Initializes the archive with random individuals and evaluates them.
        """
        init_indiv = self.random_individuals(self.k)
        init_indiv[0, :] = self.x0
        self.archive.initialize(init_indiv)
        self.evaluate_archive()
        self.archive.sort_memory()
        self.archive.build_weights()
        self.archive.build_categorical_weights(self.idx_cat, bounds=self.bounds)

    def evaluate_archive(self, index=None):
        """
            Evaluates the stored solutions in the archive.
        """
        if index is None:
            self.archive.evaluate_all_memory(self.f)
        else:
            return NotImplementedError

    def random_individuals(self, n):
        """
            Generates random individuals for initialization.

            Parameters:
            - n: Number of individuals to generate.

            Returns:
            - indiv: Generated individuals.
        """
        indiv = np.zeros((n, self.ndim))
        for idx in self.idx_cont:
            lb, ub = self.bounds[idx]
            indiv[:, idx] = np.random.uniform(lb, ub, n)
        for idx in self.idx_ord:
            lb, ub = self.bounds[idx]
            indiv[:, idx] = np.random.randint(lb, ub + 1, size=n)
        for idx in self.idx_cat:
            domain = np.array(self.bounds[idx])  # OK !! this is the domain, not bounds for categorical
            indiv[:, idx] = np.random.choice(domain, size=n)
        return indiv

    def get_best_y(self):
        """
           Returns the best function values found.
        """
        return self.best_res

    def get_best_x(self):
        """
           Returns the best solutions found.
        """
        return self.x_best_res
