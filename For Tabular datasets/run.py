
import subprocess
import re
import sys
import time

# List of seeds to run
SEEDS = [1234,123,12,1,42]
# SEEDS = [1234]

# Path to config file
CONFIG_FILE = 'config.py'

def update_config(seed, non_iid, noise):
    """
    Updates the config.py file with the given seed, non_iid, and noise values.
    """
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    # Update SEED
    content = re.sub(r'^SEED\s*=\s*\d+', f'SEED = {seed}', content, flags=re.MULTILINE)
    
    # Update non_iid
    content = re.sub(r'^non_iid\s*=\s*\d+', f'non_iid = {non_iid}', content, flags=re.MULTILINE)
    
    # Update noise
    content = re.sub(r'^noise\s*=\s*\d+', f'noise = {noise}', content, flags=re.MULTILINE)

    with open(CONFIG_FILE, 'w') as f:
        f.write(content)
    
    print(f"Updated config: SEED={seed}, non_iid={non_iid}, noise={noise}")

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
    current_run = 0
    
    start_time = time.time()

    for seed in SEEDS:
        # Configuration 1: non_iid = 0, noise = 0
        update_config(seed, 0, 0)
        current_run += 1
        print(f"\n--- Run {current_run}/{total_runs} ---")
        run_main()

        # Configuration 2: non_iid = 1, noise = 0
        update_config(seed, 1, 0)
        current_run += 1
        print(f"\n--- Run {current_run}/{total_runs} ---")
        run_main()

        # Configuration 3: non_iid = 1, noise = 1
        update_config(seed, 1, 1)
        current_run += 1
        print(f"\n--- Run {current_run}/{total_runs} ---")
        run_main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nAll experiments completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
