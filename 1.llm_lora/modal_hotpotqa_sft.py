# modal_hotpotqa_sft.py
# Minimal Modal script for HotpotQA SFT with Unsloth.

import modal

app = modal.App("hotpotqa-sft")

# Use your working versions from Colab / pip freeze
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "accelerate==1.10.0",
    "datasets==3.6.0",
    "hf_transfer==0.1.9",
    "huggingface_hub==0.34.4",
    "peft==0.17.0",
    "transformers==4.55.2",
    "trl==0.21.0",
    "unsloth==2025.8.9",
    "unsloth_zoo==2025.8.8",
)

GPU_TYPE = "A100-80GB"  # adjust as needed

with image.imports():
    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTTrainer, SFTConfig


@app.function(image=image, gpu=GPU_TYPE, timeout=60 * 60 * 6)
def finetune():
    # 1. Load model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 2. Data
    print("Loading data...")
    ds = load_dataset("rshn-krn/hotpotqa-agent-training-data-2", split="train")

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    def is_valid(example):
        try:
            tokenizer.apply_chat_template(
                example["conversations"], tokenize=False, add_generation_prompt=False
            )
            return True
        except TypeError:
            return False

    ds = ds.filter(is_valid)

    def format_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=False
            )
            for c in examples["conversations"]
        ]
        return {"text": texts}

    ds = ds.map(format_func, batched=True, num_proc=2)

    # 3. Trainer
    print("Setting up trainer...")
    cfg = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # use num_train_epochs = 1 for full run, max_steps = 5 for test run
        num_train_epochs=1,
        # max_steps=5,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="/tmp/outputs",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=cfg,
        dataset_text_field="text",
        dataset_num_proc=2,
        packing=False,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # 4. Train
    print("Starting training…")
    stats = trainer.train()
    print(stats)

    # 5. Example Prediction
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    messages = [
        {
            "content": "Your input fields are:\n1. `question` (str): \n2. `trajectory` (str):\nYour output fields are:\n1. `next_thought` (str): \n2. `next_tool_name` (Literal['evaluate_math', 'search_wikipedia', 'search_web', 'finish']): \n3. `next_tool_args` (dict[str, Any]):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## trajectory ## ]]\n{trajectory}\n\n[[ ## next_thought ## ]]\n{next_thought}\n\n[[ ## next_tool_name ## ]]\n{next_tool_name}        # note: the value you produce must exactly match (no extra characters) one of: evaluate_math; search_wikipedia; search_web; finish\n\n[[ ## next_tool_args ## ]]\n{next_tool_args}        # note: the value you produce must adhere to the JSON schema: {\"type\": \"object\", \"additionalProperties\": true}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`.\n        \n        You are an Agent. In each episode, you will be given the fields `question` as input. And you can see your past trajectory so far.\n        Your goal is to use one or more of the supplied tools to collect any necessary information for producing `answer`.\n        \n        To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.\n        After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n        \n        When writing next_thought, you may reason about the current situation and plan for future steps.\n        When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n        \n        (1) evaluate_math, whose description is <desc>Evaluate a mathematical expression safely        Args:          expression (str): The mathematical expression to evaluate        Returns:          float: The result of the evaluation or an error message      </desc>. It takes arguments {'expression': {'type': 'string'}}.\n        (2) search_wikipedia, whose description is <desc>Search Wikipedia abstracts with a given query        Args:          query (str): The search query        Returns:          List[str]: A list of Wikipedia article abstracts      </desc>. It takes arguments {'query': {'type': 'string'}}.\n        (3) search_web, whose description is <desc>      Performs a web search using the Tavily API and returns a list of search results.        Args:          query (str): The search query string.          max_results (int): The maximum number of search results to return.        Returns:          list[dict]: A list of dictionaries, each containing:              - 'title' (str): The title of the search result.              - 'snippet' (str): A snippet or content summary of the result.              - 'url' (str): The URL of the search result (truncated to 50 characters).      </desc>. It takes arguments {'query': {'type': 'string'}, 'max_results': {'type': 'integer'}}.\n        (4) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `answer`, are now available to be extracted.</desc>. It takes arguments {}.\n        When providing `next_tool_args`, the value inside the field must be in JSON format",
            "role": "system",
        },
        {
            "content": "[[ ## question ## ]]\nThe city where the Anubis Shrine was found was known to the ancient Egyptians as what?\n\n[[ ## trajectory ## ]]\n[[ ## thought_0 ## ]]\nThe user is asking for the ancient Egyptian name of the city where the Anubis Shrine was found. I need to first identify the location where the Anubis Shrine was found. I will use search_wikipedia for this.\n\n[[ ## tool_name_0 ## ]]\nsearch_wikipedia\n\n[[ ## tool_args_0 ## ]]\n{\"query\": \"Anubis Shrine discovery location\"}\n\n[[ ## observation_0 ## ]]\n[1] «Anubis Shrine | The Anubis Shrine was part of the grave gods of Tutankhamun (18th Dynasty, New Kingdom). The tomb (KV62) was discovered almost intact on 4 November 1922 in the Valley of the Kings in west Thebes by Howard Carter. Today the object, with the find number 261, is an exhibit at the Egyptian Museum in Cairo, with the inventory number JE 61444.»\n[2] «Eremiaphila anubis | Eremiaphila anubis, common name Anubis mantis, is a species of praying mantis found in Egypt.»\n[3] «Eremiaphila zolotarevskyi | Eremiaphila zolotarevskyi, common name Anubis mantis, is a species of praying mantis found in Chad.»\n\nRespond with the corresponding output fields, starting with the field `[[ ## next_thought ## ]]`, then `[[ ## next_tool_name ## ]]` (must be formatted as a valid Python Literal['evaluate_math', 'search_wikipedia', 'search_web', 'finish']), then `[[ ## next_tool_args ## ]]` (must be formatted as a valid Python dict[str, Any]), and then ending with the marker for `[[ ## completed ## ]]`.",
            "role": "user",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs, max_new_tokens=200, use_cache=True, temperature=1.5, min_p=0.1
    )
    decoded_outputs = tokenizer.batch_decode(outputs)
    print(decoded_outputs)

    # 6. Save
    print("Pushing model to hub...")
    model.push_to_hub_merged(
        "rshn-krn/hotpotqa-agent-sft-llm",
        tokenizer,
        commit_message="full training run",
        save_method="merged_16bit",
        token="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",
    )
    print("Pushed to hub!")


@app.local_entrypoint()
def main():
    finetune.remote()
