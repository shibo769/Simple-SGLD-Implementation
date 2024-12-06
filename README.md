# Simple-SGLD-Implementation

## Overview
This repository contains an implementation of the Stochastic Gradient Langevin Dynamics (SGLD) algorithm. SGLD is a stochastic optimization method that bridges traditional optimization and Bayesian posterior sampling. By adding Gaussian noise to gradient updates, SGLD transitions seamlessly from optimization to posterior sampling, making it a powerful tool for Bayesian inference on large datasets.

This implementation is inspired by the paper "Bayesian Learning via Stochastic Gradient Langevin Dynamics" by Max Welling and Yee Whye Teh (2011).

## Contents
- **`SGLD Logistic.py`**: Python implementation of SGLD and SGD algorithms, applied to a logistic regression example.
- **`WelTeh2011a.pdf`**: Reference paper for SGLD.
- **`README.md`**: This documentation.

## Features
- Implements both **SGLD** and **SGD** algorithms.
- Simulates logistic regression data.
- Compares SGLD and SGD in terms of convergence and parameter estimation.
- Provides visualization of parameter traces and convergence metrics.

## Key Methods
1. **SGLD**
    - Combines stochastic gradient descent with Langevin dynamics by injecting noise into gradient updates.
    - Approximates posterior sampling as step size anneals to zero.
2. **SGD**
    - Standard stochastic gradient descent for comparison.

## Prerequisites
- Python 3.x
- Libraries: `scipy`, `numpy`, `matplotlib`, `pandas`, `autograd`, `sklearn`

Install the required libraries using the following command:
```bash
pip install -r requirements.txt
```

## Usage
### Running the Script
1. Clone the repository:
   ```bash
   git clone https://github.com/shibo769/Simple-SGLD-Implementation.git
   cd Simple-SGLD-Implementation
   ```
2. Run the Python script:
   ```bash
   python SGLD\ Logistic.py
   ```

### Outputs
The script will:
- Simulate logistic regression data.
- Optimize parameters using SGLD and SGD.
- Plot parameter traces and convergence metrics.

## Example Code Snippets
Below is a brief snippet illustrating how SGLD is implemented:
```python
def SGLD(n_steps, burn_in, batch_size, theta_init, gamma, X, Y, thin=1):
    theta = theta_init.copy()
    thetas = []
    for i in range(n_steps):
        X_batch, Y_batch = batch(X, Y, len(X), batch_size)
        proposal_loc = 0.5 * gamma * (grad_prior(theta) + (len(X) / batch_size) * grad_lik(theta, X_batch, Y_batch))
        theta = theta + stats.multivariate_normal.rvs(mean=proposal_loc, cov=gamma)
        if i >= burn_in and i % thin == 0:
            thetas.append(theta.copy())
    return np.array(thetas)
```

## Results
- **Visualization**: The script generates trace plots comparing SGLD and SGD parameter updates.
- **Performance**: SGLD provides posterior samples while SGD converges to the maximum a posteriori (MAP) estimate.

## References
- Max Welling and Yee Whye Teh. "Bayesian Learning via Stochastic Gradient Langevin Dynamics." Proceedings of the 28th International Conference on Machine Learning (ICML), 2011.

## Acknowledgments
This work is based on the foundational research by Max Welling and Yee Whye Teh. The implementation draws upon their theoretical insights and examples from their paper.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

