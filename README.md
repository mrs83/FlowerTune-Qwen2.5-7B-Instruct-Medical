# FlowerTune LLM on Medical Dataset

This directory conducts federated instruction tuning with a pretrained [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model on a [Medical dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

## PEFT Adapter

The fine-tuning results have been submitted as a PEFT adapter and can be accessed here:

- [FlowerTune-Qwen2.5-7B-Instruct-Medical-PEFT](https://huggingface.co/mrs83/FlowerTune-Qwen2.5-7B-Instruct-Medical-PEFT)

## Methodology

This experiment performs federated LLM fine-tuning with [LoRA](https://arxiv.org/pdf/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with FedProx strategy.

### Qwen2.5-7B-Instruct

For the **Qwen-2.5 7B Instruct** model, we adopted the following fine-tuning methodology:

- **Precision**: bf16 for model weights, tf32 for gradients and optimizer states.
- **Quantization**: 4-bit quantization for reduced memory usage.
- **Optimizer**: Paged AdamW 8-bit for effective optimization under constrained resources.
- **LoRA Configuration**:
  - Rank (r): 8
  - Alpha: 32
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training Configuration**:
  - Batch size: 16
  - Maximum number of steps: 6
  - Warmup steps: 2
  - Total number of rounds: 100
  - Fraction fit per round: 0.15
- **Learning Rate Scheduler**: Constant learning rate scheduler with warmup steps, where:
  - Maximum LR: 5e-5
  - Minimum LR: 1e-6
- **Strategy**: FedProx

When bf16 and tf32 are enabled, model weights are stored in bf16 format, while gradients are computed in half-precision and converted to full 32-bit precision for updates.

### Training Loss Visualization

Below is the training loss plot from the experiment:

![Training Loss](flowertune-eval-medical/benchmarks/train_loss.png)

This methodology enabled efficient fine-tuning within constrained resources while ensuring competitive performance.

### Evaluation Results

- **pubmedqa**: 0.446
- **medqa**: 0.5546
- **medmcqa**: 0.3906
- **average**: 0.4637

### Communication Budget

46228 Megabytes

## Environments setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
pip install -e .
```

## Experimental setup

The dataset is divided into 20 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.15) of the total nodes to participate in each round, for a total of `100` rounds.
All settings are defined in `pyproject.toml`.

> [!IMPORTANT]
> Please note that `[tool.flwr.app.config.static]` and `options.num-supernodes` under `[tool.flwr.federations.local-simulation]` are not allowed to be modified for fair competition if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).


## Running the challenge

Run the challenge with default config values.
The configs are defined in `[tool.flwr.app.config]` entry of `pyproject.toml`, and are loaded automatically.

```bash
flwr run
```

## VRAM consumption

We use Mistral-7B model with 4-bit quantization as default. The estimated VRAM consumption per client for each challenge is shown below:

| Challenges | GeneralNLP |   Finance  |   Medical  |    Code    |
| :--------: | :--------: | :--------: | :--------: | :--------: |
|    VRAM    | ~25.50 GB  | ~17.30 GB  | ~22.80 GB  | ~17.40 GB  |

You can adjust the CPU/GPU resources you assign to each of the clients based on your device, which are specified with `options.backend.client-resources.num-cpus` and `options.backend.client-resources.num-gpus` under `[tool.flwr.federations.local-simulation]` entry in `pyproject.toml`.


## Model saving

The global PEFT model checkpoints are saved every 5 rounds after aggregation on the sever side as default, which can be specified with `train.save-every-round` under [tool.flwr.app.config] entry in `pyproject.toml`.

> [!NOTE]
> Please provide the last PEFT checkpoint if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).
