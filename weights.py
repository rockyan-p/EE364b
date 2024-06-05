import json
import numpy as np

def process_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    arr = np.array(data['chain_pair_pae_min'])
    return arr[0, 1], arr[1, 0]

def compute_w(json_files):
    b_vals, c_vals = [], []

    for file_path in json_files:
        b, c = process_json(file_path)
        b_vals.append(b)
        c_vals.append(c)

    return np.array(b_vals), np.array(c_vals)

def get_w():
    json_files = [
        'fold_mtb7_mta2_ec_d1_4_reglyco_summary_confidences_0.json',
        'fold_mtb7_mta3_ec_d1_4_reglyco_summary_confidences_0.json',
        'fold_mtb7_mta4_ec_d1_4_reglyco_summary_confidences_0.json',
        'fold_mtb7_mta5_ec_d1_4_reglyco_summary_confidences_0.json',
        'fold_mtb7_mta6_ec_d1_4_reglyco_summary_confidences_0.json',
        'fold_mtb7_mta7_ec_d1_4_reglyco_summary_confidences_0.json',
        'fold_mtb7_mtb7_ec_d1_4_reglyco_summary_confidences_0.json'
    ]

    w_b, w_c = compute_w(json_files)
    print(" (w_b):", w_b)
    print(" (w_c):", w_c)
    w_b /= np.sum(w_b)
    w_c /= np.sum(w_c)
    return w_b, w_c

if __name__ == "__main__":
    w_b, w_c = get_w()
