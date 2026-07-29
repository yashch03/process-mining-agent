"""
DPO fine-tuning using Unsloth + TRL on the generated preference pairs.
Section 8.7 of the implementation doc.

HONEST SCOPE NOTE: This runs on a small (9-pair) proof-of-concept dataset
to verify the training pipeline works end-to-end. Not a claim of a
robustly-trained model — see the writeup's limitations section.
"""
import json
import yaml
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from datasets import Dataset


def load_preference_dataset(path="phase4_agent/dpo/preference_pairs.json"):
    with open(path) as f:
        raw_pairs = json.load(f)

    formatted = [
        {
            "prompt": pair["context"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        }
        for pair in raw_pairs
    ]
    return Dataset.from_list(formatted)


if __name__ == "__main__":
    with open("configs/seeds.yaml") as f:
        seeds_cfg = yaml.safe_load(f)
    seed = seeds_cfg["dpo"]

    print("Loading preference dataset...")
    dataset = load_preference_dataset()
    print(f"Loaded {len(dataset)} preference pairs")

    print("Loading base model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    dpo_config = DPOConfig(
        output_dir="./dpo_checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=5e-6,
        beta=0.1,
        seed=seed,
        report_to="none",
        logging_steps=1,
    )

    print("Starting DPO training...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    print("\n✅ DPO training completed successfully")
    print("NOTE: trained on 9 pairs (proof-of-concept scale) — see writeup limitations section")

    model.save_pretrained("./dpo_final_model")
    tokenizer.save_pretrained("./dpo_final_model")
    print("Model saved to ./dpo_final_model")
