import os
import subprocess

def run_mlx_finetune():
    """
    Wrapper script to trigger Apple MLX LoRA Fine-tuning.
    We are fine-tuning a fast instruction model (Qwen 0.5B)
    to become our Domain Expert using the JSONL dataset.
    """
    # We use a very small, fast model for the initial test. 
    # For production on an M5, you could easily use a 7B or 8B model!
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    data_dir = "data"
    output_dir = "adapters"
    
    print("=" * 60)
    print("🚀 Starting Apple MLX LoRA Fine-Tuning on M5 Silicon...")
    print(f"Model: {model_name}")
    print(f"Data Directory: {data_dir} (contains train.jsonl and valid.jsonl)")
    print("=" * 60)
    
    # MLX-LM LoRA training command
    command = [
        "python", "-m", "mlx_lm.lora",
        "--model", model_name,
        "--train",
        "--data", data_dir,
        "--batch-size", "1",      # Batch size
        "--lora-layers", "8",     # How many layers to apply LoRA to
        "--iters", "100",         # Quick test run. Use 1000+ for real training.
        "--adapter-path", output_dir
    ]
    
    try:
        subprocess.run(command, check=True)
        print("\n✅ Training Complete!")
        print(f"The LoRA adapter weights have been saved to '{output_dir}'.")
        print("\nTo test your newly trained Domain Expert, run:")
        print(f"python -m mlx_lm.generate --model {model_name} --adapter-path {output_dir} --prompt 'How do I structure FastAPI?'")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")

if __name__ == "__main__":
    run_mlx_finetune()
