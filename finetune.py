import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ==========================================
# Qwen2-VL-2B LoRA Fine-Tuning Script
# ==========================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR = "./qwen2-vl-lora-output"

def main():
    print("Loading Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Loading Model in 4-bit...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True, # BitsAndBytes integration
    )

    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)

    # LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ==========================================
    # DATASET PREPARATION
    # ==========================================
    # Replace 'your-org/purchase_orders' with your actual dataset
    # Expected format: {"image": <PIL.Image>, "text": "Expected JSON extraction string"}
    print("Loading Dataset...")
    try:
        dataset = load_dataset("your-org/purchase_orders", split="train")
    except:
        print("Note: Using placeholder dataset loading. Please provide a valid dataset.")
        return

    def formatting_prompts_func(examples):
        texts = []
        for text, image in zip(examples["text"], examples["image"]):
            # Standard Qwen2-VL prompt structure for extraction
            prompt = f"<|im_start|>system\nYou are an intelligent document extraction assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Extract all Purchase Order data from this page into JSON.<|im_end|>\n<|im_start|>assistant\n{text}<|im_end|>"
            texts.append(prompt)
        return {"text": texts}

    # ==========================================
    # TRAINING ARGUMENTS
    # ==========================================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        max_steps=150, # Short training run for LoRA
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        fp16=True,
    )

    # ==========================================
    # TRAINER INITIALIZATION
    # ==========================================
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=processor.tokenizer,
        args=training_args,
    )

    print("Starting Fine-Tuning...")
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("Fine-tuning complete. You can now merge this adapter and convert to GGUF using llama.cpp.")

if __name__ == "__main__":
    main()
