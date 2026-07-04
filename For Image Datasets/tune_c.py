import os
import copy
import time
import importlib
import numpy as np
import pandas as pd

import config  # original config file

from Base.utils.utils import setup_logger, set_seed, LinearSVM_OVO
from Base.utils.load_data import load_datasets
from Base.UniCSL.unicsl_static import main_unicsl as main_unicsl_static


# ─────────────────────────────────────────────────────────────
# 🔁 Runtime config override
# ─────────────────────────────────────────────────────────────
def override_config(seed, non_iid_val, corrupt_val):
    """
    Override config variables dynamically and reload dependent modules.
    """
    config.SEED = seed
    config.non_iid = non_iid_val
    config.corrupt_data = corrupt_val

    # reset seed everywhere
    set_seed(seed)

    # reload modules that depend on config
    import Base.utils.load_data as load_data_module
    import Base.UniCSL.unicsl_static as unicsl_static_module

    importlib.reload(load_data_module)
    importlib.reload(unicsl_static_module)

    print(f"[CONFIG] SEED={seed} | non_iid={non_iid_val} | corrupt_data={corrupt_val}")


# ─────────────────────────────────────────────────────────────
# 📊 Append results to Excel (no overwrite)
# ─────────────────────────────────────────────────────────────
def append_to_excel(rows, path):
    new_df = pd.DataFrame(rows)

    if os.path.exists(path):
        old_df = pd.read_excel(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    os.makedirs(os.path.dirname(path), exist_ok=True)
    combined.to_excel(path, index=False)
    print(f"[EXCEL] Results appended → {path}")


# ─────────────────────────────────────────────────────────────
# ⚙️ Settings
# ─────────────────────────────────────────────────────────────
C_LIST = [0.01, 0.1, 1, 10, 20, 30]

SEEDS = [1, 12, 123, 1234, 42]
SCENARIOS = [(0, 0), (1, 0), (1, 1)]  # (non_iid, corrupt_data)

OUTPUT_FILE = "results/c_tuning_combined.xlsx"


# mapping algorithms
ALGO_RUNNERS = {
    'unicsl_static': lambda gp, tr, vl, te, cs, nc, c:
        main_unicsl_static(gp, tr, vl, te, cs, nc, c)[0],
}

# logger
log_file, tee = setup_logger(save_log=config.save_log)

# store all rows
all_rows = []


# ─────────────────────────────────────────────────────────────
# 🚀 Main experiment loop
# ─────────────────────────────────────────────────────────────
for seed in SEEDS:
    for non_iid_val, corrupt_val in SCENARIOS:

        # override config globally
        override_config(seed, non_iid_val, corrupt_val)

        # regenerate client schedule AFTER seed set
        client_schedule = [
            np.random.choice(config.num_clients, size=config.participants, replace=False)
            for _ in range(config.CR)
        ]

        for dataset in config.all_datasets:

            print(f"\n[DATASET] {dataset}")

            # load dataset after config override
            train_data_list, val_data_list, global_test_data = load_datasets(dataset)
            # continue
            # dataset specs
            if dataset in ['mnist', 'fashionmnist']:
                num_classes, num_features = 10, 784
            elif dataset == 'cifar10':
                num_classes, num_features = 10, 2048
            elif dataset == 'femnist':
                num_classes, num_features = 62, 784
            else:
                raise ValueError(f"Unknown dataset {dataset}")

            global_parameters = LinearSVM_OVO(num_features, num_classes).to(config.device)

            for aggregation in config.aggregations:

                if aggregation not in ALGO_RUNNERS:
                    print(f"[WARN] Unknown aggregation '{aggregation}', skipping.")
                    continue

                runner = ALGO_RUNNERS[aggregation]

                # 🧾 one row per (seed, dataset, scenario)
                row = {
                    "seed": seed,
                    "dataset": dataset,
                    "scenario": f"{non_iid_val},{corrupt_val}"
                }

                for c in C_LIST:
                    print(f"[RUN] seed={seed} dataset={dataset} scenario=({non_iid_val},{corrupt_val}) c={c}")

                    try:
                        result = runner(
                            copy.deepcopy(global_parameters),
                            copy.deepcopy(train_data_list),
                            copy.deepcopy(val_data_list),
                            copy.deepcopy(global_test_data),
                            client_schedule,
                            num_classes,
                            c
                        )

                        acc_array = np.array(result[0])
                        avg_acc = float(np.mean(acc_array))
                        std_acc = float(np.std(acc_array))

                        value = f"{avg_acc:.4f} ± {std_acc:.4f}"

                        print(f"   → {value} | time={result[1]:.1f}s")

                    except Exception as e:
                        print(f"   [ERROR] {e}")
                        value = "ERROR"

                    row[f"lr {c}"] = value

                # store row
                all_rows.append(row)


# ─────────────────────────────────────────────────────────────
# 💾 Save results
# ─────────────────────────────────────────────────────────────
append_to_excel(all_rows, OUTPUT_FILE)


# ─────────────────────────────────────────────────────────────
# 🔚 Close logger
# ─────────────────────────────────────────────────────────────
if log_file:
    log_file.close()
    tee.close()

print("\n✅ ALL EXPERIMENTS COMPLETED")