import ACO
import numpy as np

'''
    StyblinskiTang bowl shaped function reworked with continuous, ordinal and categorical variables. 
'''


class function_to_optimize:

    def __init__(self):
        self.domain = [
            {
                "name": "var1",
                "type": "cont",
                "domain": [0, 1]
            },
            {
                "name": "var4",
                "type": "ord",
                "domain": [0, 18]
            },
            {
                "name": "var6",
                "type": "cat",
                "domain": [0, 1, 2, 3, 4, 5, 6, 7]
            }
        ]
        # Styblinski_tang input defined on LB, UB. Need some correspondance.

        self.lb = -5
        self.ub = 5
        self.cont_dim = []
        self.cat_dim = []
        self.ord_dim = []
        ww = 0
        for dd in self.domain:
            if dd['type'] == 'cont':
                self.cont_dim.append(ww)
            if dd['type'] == 'ord':
                self.ord_dim.append(ww)
            if dd['type'] == 'cat':
                self.cat_dim.append(ww)
            ww += 1

    def f(self, vect):
        if not isinstance(vect, np.ndarray):
            X_tmp = np.asarray(vect)
        else:
            X_tmp = vect
        if len(X_tmp.shape) == 1:
            X_tmp = X_tmp.reshape(1, -1)
        else:
            X_tmp = X_tmp
        inputs = np.zeros(X_tmp.shape)
        ww = 0
        for X in X_tmp:
            ii = 0
            for e in X:
                x01 = (e - self.domain[ii]["domain"][0]) / (
                        self.domain[ii]["domain"][1] - self.domain[ii]["domain"][0])
                inputs[ww, ii] = self.lb + x01 * (self.ub - self.lb)  #
                ii += 1
            ww += 1

        results = np.zeros(X_tmp.shape[0])
        kk = 0
        for input in inputs:
            res = 0
            ii = 0
            for element in input:
                res += element ** 4 - 16 * element ** 2 + 5 * element
                ii += 1
            results[kk] = (0.5 ** ii * res)
            kk += 1
        return results


def minimal_example():
    func = function_to_optimize()

    bounds = [dom["domain"] for dom in func.domain]
    # Bounds of the search space
    # Continuous variables, usually [0, 1]
    # Ordinal variables, usually [0, N]
    # Categorical variables, usually [0, 1, ..., M]

    # Initial random solution. Will be evaluated and put inside archive.
    x0 = [np.random.uniform(),
          np.random.randint(func.domain[1]['domain'][0], func.domain[1]['domain'][1], 1)[0],
          np.random.randint(func.domain[2]['domain'][0], func.domain[2]['domain'][1], 1)[0]
          ]

    # hyper-parameters of the colony
    parameters = {
        "m": 5,
        "q": 0.05,
        "xi": 1,  # 0.67,
        "k": 90,
        "MaxStagIter": 1000,
        "eps": 10e-5
    }

    colony = ACO.colony.Colony(bounds, parameters, 3, func.ord_dim, func.cat_dim, criterion=2000, maximize=False,
                               debug=False)

    x_opt, f_opt = colony.optimize(x0, func.f)
    print("X_opt : " + str(x_opt))
    print("Objective : " + str(f_opt))


if __name__ == '__main__':
    minimal_example()
