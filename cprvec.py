import json
import cvxpy as cp
import numpy as np
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
from weights import get_w

json_files = [
    'fold_mtb7_mta2_ec_d1_4_reglyco_full_data_0.json', 
    'fold_mtb7_mta3_ec_d1_4_reglyco_full_data_0.json', 
    'fold_mtb7_mta4_ec_d1_4_reglyco_full_data_0.json', 
    'fold_mtb7_mta5_ec_d1_4_reglyco_full_data_0.json', 
    'fold_mtb7_mta6_ec_d1_4_reglyco_full_data_0.json', 
    'fold_mtb7_mta7_ec_d1_4_reglyco_full_data_0.json', 
    'fold_mtb7_mtb7_ec_d1_4_reglyco_full_data_0.json'
]

a_size = 547 # chain size for the 'a' block
d_size = 478 # minimum chain size for the 'd' block
weights_b, weights_c = get_w()

print(weights_b)
print(weights_c)

def process_json(file_path, a_size, d_size):
    with open(file_path, 'r') as f:
        data = json.load(f)
    contact_probs = np.array(data['contact_probs'])
    a_idx, b_idx = range(a_size), range(a_size, a_size + d_size)
    return (contact_probs[np.ix_(a_idx, a_idx)], contact_probs[np.ix_(a_idx, b_idx)], 
            contact_probs[np.ix_(b_idx, a_idx)], contact_probs[np.ix_(b_idx, b_idx)])

a_blocks, b_blocks, c_blocks, d_blocks = [], [], [], []

for file_path in json_files:
    a, b, c, d = process_json(file_path, a_size, d_size)
    a_blocks.append(a)
    b_blocks.append(b)
    c_blocks.append(c)
    d_blocks.append(d)

a_mean = np.mean(a_blocks, axis=0)
n, m, lambda_reg = a_size, d_size, 0.5

b_hat = cp.Variable(n)
c_hat = cp.Variable(n)

objective = cp.Minimize(
    sum(weights_b[i] * cp.norm(np.sum(b_blocks[i], axis=1) - b_hat, 1) + 
        weights_c[i] * cp.norm(np.sum(c_blocks[i], axis=0) - c_hat, 1) for i in range(len(json_files))) +
    lambda_reg * (cp.norm(b_hat, 1) + cp.norm(c_hat, 1))
)

constraints = [b_hat == c_hat.T, b_hat >= 0, b_hat <= 1, c_hat >= 0, c_hat <= 1]
problem = cp.Problem(objective, constraints)

start_time = time.time()
problem.solve()
end_time = time.time()

print(f"Solve time: {end_time - start_time:.2f} seconds")

b_hat_value, c_hat_value = b_hat.value, c_hat.value

for i in range(len(json_files)):
    print(1 * ((np.linalg.norm(np.max(b_blocks[i], axis=1) - b_hat_value, 2))**2 + 
               1 * (np.linalg.norm(np.max(c_blocks[i], axis=0) - c_hat_value, 2))**2))

for i in range(len(json_files)):
    print('b')
    print(np.round(np.sum(b_blocks[i], axis=1) - b_hat_value, 1))

top_indices = np.argsort(b_hat_value)[-10:][::-1]
print("\nTop 10 values and indices:")
for index in top_indices:
    print(f"Index: {index+1}, Value: {b_hat_value[index]}")

plt.figure(figsize=(10, 6))
plt.hist(b_hat_value, bins=50, log=True)
plt.xlabel('Contact Probability')
plt.ylabel('Frequency (log scale)')
plt.savefig('b_hat_histogram.png')

def plot_histogram(blocks, hat_value):
    combined_residuals = np.concatenate([np.sum(block, axis=0) - hat_value for block in blocks])
    fig, ax = plt.subplots(figsize=(5, 10))
    ax.hist(combined_residuals, bins=50, range=(-1, 1), edgecolor='black', label='Residuals')
    ax.set_title(r'$\sum_i \mathbf{b}_i - \hat{\mathbf{b}}$')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('Combined_Residuals_Histogram.png')

plot_histogram(c_blocks, c_hat_value)
