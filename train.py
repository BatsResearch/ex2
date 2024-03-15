import gc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.import_utils import is_xpu_available

import image_text_reward
import load_image_dataset
import utils

tqdm.pandas()


@dataclass
class LocalPPOConfig:
    model_name: Optional[str] = "mistralai/Mistral-7B-v0.1"
    steps: int = 1000
    learning_rate: float = 1.41e-5
    log_with: Optional[str] = None
    tracker_project_name: Optional[str] = "rl-training"
    batch_size: int = 32
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    optimize_cuda_cache: Optional[bool] = True
    early_stopping: Optional[bool] = False
    target_kl: float = 0.1
    target: float = 10
    ppo_epochs: int = 4
    seed: int = 0
    init_kl_coef: float = 0.2
    adap_kl_ctrl: Optional[bool] = True
    use_score_scaling: Optional[bool] = False
    use_score_norm: Optional[bool] = False


@dataclass
class LocalLoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class ScriptArguments:
    output_dir: str
    """Where to save the model checkpoints"""
    data_root: str
    """Image dataset data root"""
    model_name: Optional[str] = None
    """Name of the model"""
    dataset_name: Optional[str] = None
    """Name of the image dataset"""
    split: Optional[str] = None
    """Image dataset split"""
    vlm_name: Optional[str] = "openai/clip-vit-base-patch32"
    """VLM model name"""
    vlm_16bit: Optional[str] = "True"
    """Whether to load the vlm in 16 bit"""
    num_images_for_reward: Optional[int] = 256
    """Number of images per class used to calculate the reward"""
    save_freq: Optional[int] = None
    """n steps to save the model"""
    ppo_config: LocalPPOConfig = field(default_factory=lambda: LocalPPOConfig())
    lora_config: LocalLoraConfig = field(default_factory=lambda: LocalLoraConfig())


args: ScriptArguments = tyro.cli(ScriptArguments)

if args.ppo_config.log_with == "none":
    args.ppo_config.log_with = None

args.model_name = args.model_name or args.ppo_config.model_name
args.ppo_config.model_name = args.model_name or args.ppo_config.model_name
args.ppo_config = PPOConfig(**asdict(args.ppo_config))
args.lora_config = LoraConfig(**asdict(args.lora_config))


def build_dataset(tokenizer, queries):
    ds = Dataset.from_dict({"query_text": queries})
    original_columns = ds.column_names
    num_proc = 8

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for query in examples["query_text"]:
            tokenized_query = tokenizer(query, truncation=True, max_length=2048)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_query["input_ids"])

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name, revision=None)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading FT query dataset")
(
    raw_queries,
    query_to_metadata_map,
    ft_class_queries,
) = utils.single_class_text_dataset(args.dataset_name)
dataset = build_dataset(tokenizer, queries=raw_queries)


train_gen_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "min_length": -1,
    "eos_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}


set_seed(args.ppo_config.seed)

current_device = Accelerator().local_process_index
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    args.ppo_config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=args.lora_config,
)

print("Instantiate PPO trainer")
ppo_trainer = PPOTrainer(
    args.ppo_config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=None,
)


print("Instantiate Reward")
reward_calculator = image_text_reward.ImageTextReward()

print("Load VLM")
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"

_vlm_class = utils.get_vlm_class(vlm_name=args.vlm_name)
vlm = _vlm_class(vlm_name=args.vlm_name)
vlm.load_processor()

gc.collect()
if is_xpu_available():
    torch.xpu.empty_cache()
else:
    torch.cuda.empty_cache()

vlm.load_model(device=device, load_in_16bit=args.vlm_16bit)


print("Load FT Image Dataset")
image_dataset, per_cls_indices = load_image_dataset.get_vision_dataset(
    config={
        "root": Path(args.data_root).as_posix(),
        "dataset": args.dataset_name,
        "split": args.split,
    },
    per_cls_indices=True,
    num_images_per_cls=args.num_images_for_reward,
)


curr_step = -1
curr_epoch = -1

with tqdm(total=args.ppo_config.steps) as pbar:
    while True:
        if curr_step + 1 >= args.ppo_config.steps:
            break
        curr_epoch += 1
        for batch in ppo_trainer.dataloader:
            if curr_step + 1 >= args.ppo_config.steps:
                break
            curr_step += 1

            question_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                generate_ref_response=False,
                batch_size=16,
                length_sampler=None,
                remove_padding=True,
                **train_gen_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )
            with torch.no_grad():
                rewards = reward_calculator.calc(
                    queries=batch["query"],
                    responses=batch["response"],
                    query2metadata=query_to_metadata_map,
                    img_dataset=image_dataset,
                    per_cls_indices=per_cls_indices,
                    vlm=vlm,
                )
            rewards = [torch.tensor(r) for r in rewards]

            batch["ref_response"] = ["-"] * len(batch["response"])
            batch["ref_rewards"] = [torch.tensor(0) for _ in batch["response"]]

            # Run PPO step
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(
                stats,
                batch,
                rewards,
                columns_to_log=["query", "response", "ref_response", "ref_rewards"],
            )

            if args.save_freq and curr_step != 0 and curr_step % args.save_freq == 0:
                ppo_trainer.accelerator.wait_for_everyone()
                if ppo_trainer.accelerator.is_main_process:
                    path_ = (
                        Path(args.output_dir).joinpath(f"step_{curr_step}").as_posix()
                    )

                    ppo_trainer.save_pretrained(
                        path_,
                        commit_message=f"step {curr_step}",
                        push_to_hub=False,
                    )
            pbar.update(1)


ppo_trainer.accelerator.wait_for_everyone()
if ppo_trainer.accelerator.is_main_process:
    path_ = Path(args.output_dir).joinpath(f"step_{curr_step}")
    if not path_.exists():
        ppo_trainer.save_pretrained(
            path_,
            commit_message=f"step {curr_step}",
            push_to_hub=False,
        )
