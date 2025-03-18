## Python implementation of Ant colony optimizaiton for mixed-variable optimizaiton problems - Lian T. et al.


### How to use 
- Requirements in requirements.txt
- Python used for development 3.8.x
- Launch commands
```
python3 setup.py build
python3 setup.py install
```
- colony.optimize is the primary interface of the optimizer (example in example.py).

### More info

- Bounding constraints on continuous variables are implemented using a truncated normal distribution which can be long to compute.

### TODO 
- Example with the 3 variable  types
- Remove maximize. Always minimize.

### References

[Liao, Tianjun, et al. "Ant colony optimization for mixed-variable optimization problems."](https://doi.org/10.1109/TEVC.2013.2281531) *IEEE Transactions on Evolutionary Computation* 18.4 (2013): 503-518.
