# Plan: Two-Step Fine-tuning Notebook

## Overview

Create `finetune-twostep.py` - a marimo notebook for training the two-step coffee order parsing pipeline:

1. **Simplification model**: messy order text → clean structured text
2. **Structured-output model**: clean text → JSON

## Key Insight

The structured-output model's training data depends on the simplification model's outputs. This creates a pipeline where:
- Step 1 training uses static ground truth
- Step 2 training uses dynamically generated inputs from the trained Step 1 model

## Notebook Structure

### 1. Imports & Setup
Same as `finetune.py` (torch, transformers, peft, etc.) with device detection.

### 2. TwoStepParams Configuration

```python
class TwoStepParams(BaseModel):
    model: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    num_epochs_step1: int = Field(default=100, description="Epochs for simplification model")
    num_epochs_step2: int = Field(default=100, description="Epochs for structured-output model")
    n_examples: int = Field(default=50)
```

### 3. Load Training Data

- `input/train.csv` (order, expected_json)
- `input/train_simplified.csv` (order, simplified) - ground truth for Step 1

### 4. Load System Prompts

- `input/prompts/simplification/v1.md` for Step 1
- `input/prompts/structured-output/v1.md` for Step 2

### 5. Step 1: Train Simplification Model

**Training data:**
- Input: `row["order"]` (messy text)
- Output: `row["simplified"]` (clean text)

**Output:** Adapter saved to `outputs/adapters-simplification/`

### 6. Generate Simplified Text

Run trained Step 1 model on all orders:

```python
for row in train_df:
    simplified = run_simplification_model(row["order"])
    generated.append(simplified)
```

Save to `outputs/generated_simplified.csv`.

### 7. Step 2: Train Structured-Output Model

**Training data:**
- Input: generated simplified text (from Step 6)
- Output: `row["expected_json"]`

**Output:** Adapter saved to `outputs/adapters-structured/`

### 8. Evaluation

Compare on test examples:
- Order (original)
- Simplified (Step 1 output)
- Predicted JSON (Step 2 output)
- Expected JSON (ground truth)
- Price validation

## Data Flow

```
train.csv                     train_simplified.csv
    │                                │
    │ order                          │ simplified (ground truth)
    ▼                                ▼
┌─────────────────────────────────────────┐
│  Step 1: Train Simplification Model     │
│  input: order → output: simplified      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Generate: Run Step 1 on all orders     │
│  produces: generated_simplified.csv     │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Step 2: Train Structured-Output Model  │
│  input: simplified → output: JSON       │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Evaluate: Full pipeline                │
│  order → Step1 → simplified → Step2 → JSON
└─────────────────────────────────────────┘
```

## CLI Usage

```bash
uv run marimo run finetune-twostep.py -- \
  --n-examples 100 \
  --num-epochs-step1 200 \
  --num-epochs-step2 150
```

## Files

```
hong-kong/
├── finetune.py              # Existing: end-to-end training
├── finetune-twostep.py      # New: two-step pipeline
├── input/
│   ├── train.csv
│   ├── train_simplified.csv # Needs creation
│   └── prompts/
│       ├── simplification/v1.md
│       └── structured-output/v1.md
└── outputs/
    ├── adapters-simplification/
    ├── adapters-structured/
    └── generated_simplified.csv
```

## Open Questions

1. **Ground truth for simplification**: How to create `train_simplified.csv`?
   - Hand-write?
   - Generate with GPT-4?

2. **Simplified text format**: Use bullet format from `kaggle-coffee.py`?
   ```
   - Venti Drip Coffee – Extra Hot
   - 3 × Trenta Chai Latte – Caramel Drizzle, No Whip
   ```

3. **Same base model for both steps?**
