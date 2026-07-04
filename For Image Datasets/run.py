
import subprocess
import re
import sys
import time

# List of seeds to run
SEEDS = [1234, 123, 12, 1, 42]
# SEEDS = [1]

# Path to config file
CONFIG_FILE = 'config.py'

def update_config(seed, non_iid, corrupt_data):
    """
    Updates the config.py file with the given seed, non_iid, and corrupt_data values.
    """
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    # Update SEED
    content = re.sub(r'^SEED\s*=\s*\d+', f'SEED = {seed}', content, flags=re.MULTILINE)
    
    # Update non_iid
    content = re.sub(r'^non_iid\s*=\s*\d+', f'non_iid = {non_iid}', content, flags=re.MULTILINE)
    
    # Update corrupt_data
    content = re.sub(r'^corrupt_data\s*=\s*\d+', f'corrupt_data = {corrupt_data}', content, flags=re.MULTILINE)

    with open(CONFIG_FILE, 'w') as f:
        f.write(content)
    
    print(f"Updated config: SEED={seed}, non_iid={non_iid}, corrupt_data={corrupt_data}")

def run_main():
    """
    Runs the main.py script.
    """
    print("Running main.py...")
    try:
        subprocess.run([sys.executable, 'main.py'], check=True)
        print("main.py finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    total_runs = len(SEEDS) * 3
    # total_runs = len(SEEDS)
    current_run = 0
    
    start_time = time.time()

    for seed in SEEDS:
        # # Configuration 1: non_iid = 0, corrupt_data = 0
        update_config(seed, 0, 0)
        current_run += 1
        print(f"\n--- Run {current_run}/{total_runs} ---")
        run_main()

        # # Configuration 2: non_iid = 1, corrupt_data = 0
        update_config(seed, 1, 0)
        current_run += 1
        print(f"\n--- Run {current_run}/{total_runs} ---")
        run_main()

        # Configuration 3: non_iid = 1, corrupt_data = 1
        update_config(seed, 1, 1)
        current_run += 1
        print(f"\n--- Run {current_run}/{total_runs} ---")
        run_main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nAll experiments completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()



# import subprocess
# import re
# import sys
# import time
# from Base.utils.utils import setup_logger

# # List of seeds to run
# # SEEDS = [1234, 123, 12, 1, 42]
# SEEDS = [1]

# # Path to config file
# CONFIG_FILE = 'config.py'

# def update_config(seed, non_iid, corrupt_data):
#     """
#     Updates the config.py file with the given seed, non_iid, and corrupt_data values.
#     """
#     with open(CONFIG_FILE, 'r') as f:
#         content = f.read()

#     # Update SEED
#     content = re.sub(r'^SEED\s*=\s*\d+', f'SEED = {seed}', content, flags=re.MULTILINE)
    
#     # Update non_iid
#     content = re.sub(r'^non_iid\s*=\s*\d+', f'non_iid = {non_iid}', content, flags=re.MULTILINE)
    
#     # Update corrupt_data
#     content = re.sub(r'^corrupt_data\s*=\s*\d+', f'corrupt_data = {corrupt_data}', content, flags=re.MULTILINE)

#     with open(CONFIG_FILE, 'w') as f:
#         f.write(content)
    
#     print(f"Updated config: SEED={seed}, non_iid={non_iid}, corrupt_data={corrupt_data}")


# def run_main(aggregation):
#     """
#     Runs main.py for a single aggregation in a fresh, isolated process.
#     Each call = clean memory slate (no BLAS/PyTorch pool contamination).
#     """
#     print(f"Running main.py for [{aggregation}]...")
#     try:
#         subprocess.run([sys.executable, 'main.py', aggregation], check=True)
#         print(f"main.py [{aggregation}] finished successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error running main.py [{aggregation}]: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")


# def main():
#     from config import aggregations  # read algorithm list from config

#     configs = [(0, 0), (1, 0), (1, 1)]  # (non_iid, corrupt_data)
#     total_runs = len(SEEDS) * len(configs) * len(aggregations)
#     current_run = 0

#     start_time = time.time()

#     for seed in SEEDS:
#         for non_iid, corrupt_data in configs:
#             update_config(seed, non_iid, corrupt_data)

#             for aggregation in aggregations:
#                 current_run += 1
#                 print(f"\n--- Run {current_run}/{total_runs} | seed={seed} non_iid={non_iid} corrupt={corrupt_data} algo={aggregation} ---")
#                 run_main(aggregation)  # <-- fresh process per algorithm

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"\nAll experiments completed in {elapsed_time:.2f} seconds.")

# if __name__ == "__main__":

#     log_file, tee = setup_logger(save_log=1)

#     main()

#     if log_file:
#         log_file.close()
#         tee.close()