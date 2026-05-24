import itertools
import subprocess
import sys

def run_benchmarks():
    """
    Sequentially launches SwarmEcho training runs for a grid of hyperparameters.
    If a run OOMs or fails, it will catch the error and continue to the next config.
    """
    
    # --- Configure Grid Here ---
    # 128, 256, 512, 1024, 2048, 4096
    num_envs_list  = [4096]
    num_steps_list = [256]      # buffer sizes
    max_steps_list = [512]          # episode lengths

    minibatches = 8
    
    # Shared settings
    total_timesteps = 100_000_000           # 1M for a quick speed/learning benchmark
    level = "00"                          # Training on Level 01
    critic_type = "global_mean"
    
    # Generate all combinations
    combinations = list(itertools.product(num_envs_list, num_steps_list, max_steps_list))
    
    print(f"🚀 Starting benchmark queue with {len(combinations)} configurations...")
    print("=" * 70)
    
    for i, (envs, steps, ep_len) in enumerate(combinations):

        # Create a clean, descriptive run name
        run_name = f"bench_env{envs}_buf{steps}_ep{ep_len}"
        
        print(f"\n[{i+1}/{len(combinations)}] Starting run: {run_name}")
        print(f"          Environments: {envs} | Buffer: {steps} | Episode: {ep_len}")
        print("-" * 70)
        
        # Build the exact CLI command
        cmd = [
            "uv", "run", "python", "src/training/train.py",
            f"level={level}",
            f"training.total_timesteps={total_timesteps}",
            f"training.num_envs={envs}",
            f"training.num_steps={steps}",
            f"env.max_steps={ep_len}",
            f"training.num_minibatches={minibatches}",  # 4M / 128 = ~32k minibatch size
            f"network.critic_type={critic_type}",
            f"logging.run_name={run_name}",
            f"logging.use_timestamp_postfix=False",
            f"logging.wandb_mode=online"
        ]
        
        try:
            # subprocess.run blocks until the child process completes
            subprocess.run(cmd, check=True)
            print(f"✅ {run_name} completed successfully.")
            
        except subprocess.CalledProcessError as e:
            # This catches OOM errors or other crashes, letting the script continue
            print(f"❌ {run_name} FAILED (exit code {e.returncode}). Likely OOM. Skipping to next...")
            
        except KeyboardInterrupt:
            # Allows you to cleanly Ctrl+C out of the entire benchmark loop
            print("\n🛑 Benchmark queue manually interrupted by user. Exiting.")
            sys.exit(0)

if __name__ == "__main__":
    run_benchmarks()
