# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.19.7",
#     "pandas>=2.0.0",
#     "datasets>=2.0.0",
#     "transformers>=4.38.0",
#     "peft>=0.10.0",
#     "torch>=2.1.0",
#     "accelerate>=0.27.0",
#     "bitsandbytes>=0.42.0; sys_platform != 'darwin'",
#     "ipython==9.10.0",
#     "wigglystuff==0.2.21",
#     "wandb==0.24.2",
#     "python-dotenv==1.2.1",
#     "pydantic>=2.0.0",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import marimo as mo
    import torch

    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    device = get_device()
    use_4bit = device == "cuda"
    print(f"Using device: {device}")
    return device, mo, os, torch, use_4bit


@app.cell
def _():
    from wigglystuff import EnvConfig
    from dotenv import load_dotenv

    load_dotenv(".env")

    env_config = EnvConfig(["WANDB_API_KEY"])
    env_config
    return (env_config,)


@app.cell
def _():
    from pydantic import BaseModel, Field, ConfigDict

    class ModelParams(BaseModel):
        num_epochs: int = Field(default=3, description="Number of training epochs.")
        n_examples: int = Field(default=5, description="Number of training examples to use.")

        model_config = ConfigDict(strict=True)
    return (ModelParams,)


@app.cell
def _(ModelParams, mo):
    model_params = ModelParams(**{k.replace("-", "_"): v for k, v in mo.cli_args()._params.items()})
    return (model_params,)


@app.cell
def _():
    import json
    import pandas as pd

    train_df = pd.read_csv("input/train.csv")
    print(f"Loaded {len(train_df)} training examples")
    return json, train_df


@app.cell
def _(mo, model_params):
    config_form = mo.ui.batch(
        mo.md("""
        **Training Configuration**

        Base Model: {model}

        Training Examples: {n_examples}

        Training Epochs: {n_epochs}
        """),
        {
            "model": mo.ui.dropdown(
                options={
                    "Qwen 2.5 3B": "Qwen/Qwen2.5-3B-Instruct",
                    "Qwen 2.5 0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
                },
                value="Qwen 2.5 0.5B",
                label="Base Model",
            ),
            "n_examples": mo.ui.slider(1, 100, value=model_params.n_examples, label="Training Examples"),
            "n_epochs": mo.ui.slider(1, 10, value=model_params.num_epochs, label="Training Epochs"),
        },
    ).form()
    config_form
    return (config_form,)


@app.cell
def _(config_form, json, train_df):
    def format_example(row) -> dict:
        return {
            "messages": [
                {"role": "system", "content": "Parse this coffee order to JSON."},
                {"role": "user", "content": row["order"]},
                {"role": "assistant", "content": row["expected_json"]},
            ]
        }

    _n_examples = config_form.value["n_examples"] if config_form.value else 5
    training_data = [format_example(row) for _, row in train_df.head(_n_examples).iterrows()]
    print(f"Formatted {len(training_data)} examples")
    print(f"Example:\n{json.dumps(training_data[0], indent=2)[:300]}...")
    return (training_data,)


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start Training")
    train_button
    return (train_button,)


@app.cell
def _(
    config_form,
    device,
    env_config,
    json,
    os,
    torch,
    train_button,
    training_data,
    use_4bit,
):
    adapter_path = "outputs/adapters"

    if train_button.value and config_form.value:
        _model_name = config_form.value["model"]

        # Configure W&B if key is available
        _use_wandb = "WANDB_API_KEY" in env_config
        if _use_wandb:
            import wandb
            os.environ["WANDB_API_KEY"] = env_config["WANDB_API_KEY"]
            wandb.init(project="barista-finetune", name=f"{_model_name.split('/')[-1]}-lora")

        # Write training data to data dir
        os.makedirs("outputs", exist_ok=True)
        _data_dir = "outputs/data"
        os.makedirs(_data_dir, exist_ok=True)
        _train_path = os.path.join(_data_dir, "train.jsonl")
        with open(_train_path, "w") as f:
            for _ex in training_data:
                f.write(json.dumps(_ex) + "\n")
        print(f"Wrote {len(training_data)} examples to {_train_path}")

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset

        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(_model_name, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        # Load model
        _model_kwargs = {"trust_remote_code": True}
        if use_4bit:
            from transformers import BitsAndBytesConfig
            _bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            _model_kwargs["quantization_config"] = _bnb_config
            _model_kwargs["device_map"] = "auto"
        else:
            _model_kwargs["torch_dtype"] = torch.float32 if device == "cpu" else torch.float16

        _model = AutoModelForCausalLM.from_pretrained(_model_name, **_model_kwargs)

        # Configure LoRA
        _lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        _model = get_peft_model(_model, _lora_config)
        _model.print_trainable_parameters()

        # Load dataset
        _dataset = load_dataset("json", data_files=_train_path, split="train")

        def _preprocess(examples):
            _texts = [
                _tokenizer.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=False)
                for _msgs in examples["messages"]
            ]
            _tok = _tokenizer(_texts, truncation=True, max_length=512, padding="max_length")
            _tok["labels"] = _tok["input_ids"].copy()
            return _tok

        _tokenized = _dataset.map(_preprocess, batched=True, remove_columns=_dataset.column_names)

        # Training
        _training_args = TrainingArguments(
            output_dir=adapter_path,
            num_train_epochs=config_form.value["n_epochs"],
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            fp16=(device == "cuda"),
            logging_steps=10,
            save_strategy="epoch",
            gradient_checkpointing=True,
            report_to="wandb" if _use_wandb else "none",
        )

        _trainer = Trainer(
            model=_model,
            args=_training_args,
            train_dataset=_tokenized,
            data_collator=DataCollatorForLanguageModeling(_tokenizer, mlm=False),
        )

        print(f"Training {_model_name} with PEFT LoRA on {device}...")
        _trainer.train()

        _model.save_pretrained(adapter_path)
        _tokenizer.save_pretrained(adapter_path)
        print(f"Training complete! Adapters saved to {adapter_path}")
    return (adapter_path,)


@app.cell
def _(mo):
    mo.md("""
    ## Compare Models
    """)
    return


@app.cell
def _(mo, train_df):
    # Pick some hard examples with corrections
    hard_examples = train_df[train_df["order"].str.contains("scratch|cancel|remove|actually|nevermind", case=False)].head(10)
    example_options = {row["order"][:60] + "...": row["id"] for _, row in hard_examples.iterrows()}

    example_dropdown = mo.ui.dropdown(
        options=example_options,
        value=list(example_options.keys())[0] if example_options else None,
        label="Select Example",
    )
    example_dropdown
    return (example_dropdown,)


@app.cell
def _(example_dropdown, json, mo, train_df):
    selected_row = train_df[train_df["id"] == example_dropdown.value].iloc[0]
    test_order = selected_row["order"]
    expected_json = json.dumps(json.loads(selected_row["expected_json"]), indent=2)

    mo.vstack([
        mo.md("**Order:**"),
        mo.md(f"_{test_order}_"),
    ])
    return expected_json, test_order


@app.cell
def _(
    adapter_path,
    config_form,
    device,
    expected_json,
    mo,
    os,
    test_order,
    torch,
):
    _model_name = config_form.value["model"] if config_form.value else "Qwen/Qwen2.5-0.5B-Instruct"
    _base_output = ""
    _finetuned_output = ""

    if test_order:
        from transformers import AutoModelForCausalLM as _AutoModel
        from transformers import AutoTokenizer as _AutoTok
        from peft import PeftModel as _PeftModel

        _messages = [
            {"role": "system", "content": "Parse this coffee order to JSON."},
            {"role": "user", "content": test_order},
        ]

        _tok = _AutoTok.from_pretrained(_model_name, trust_remote_code=True)
        _dtype = torch.float32 if device == "cpu" else torch.float16

        _base_model = _AutoModel.from_pretrained(
            _model_name,
            torch_dtype=_dtype,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            _base_model = _base_model.to(device)

        _prompt = _tok.apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
        _inputs = _tok(_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            _outputs = _base_model.generate(
                **_inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=_tok.eos_token_id,
            )
        _base_output = _tok.decode(_outputs[0][_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Finetuned model (if adapters exist)
        if os.path.exists(adapter_path):
            _ft_model = _PeftModel.from_pretrained(_base_model, adapter_path)
            _ft_model.eval()

            with torch.no_grad():
                _outputs = _ft_model.generate(
                    **_inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=_tok.eos_token_id,
                )
            _finetuned_output = _tok.decode(_outputs[0][_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        else:
            _finetuned_output = "(No adapters found - run training first)"

    mo.hstack([
        mo.vstack([
            mo.md("### Expected"),
            mo.md(f"```json\n{expected_json}\n```"),
        ]),
        mo.vstack([
            mo.md("### Base Model"),
            mo.md(f"```json\n{_base_output}\n```"),
        ]),
        mo.vstack([
            mo.md("### Finetuned"),
            mo.md(f"```json\n{_finetuned_output}\n```"),
        ]),
    ])
    return


if __name__ == "__main__":
    app.run()
