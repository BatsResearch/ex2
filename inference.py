import gc
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import tyro
from huggingface_hub.utils import SoftTemporaryDirectory
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.import_utils import is_xpu_available
from vllm import LLM, SamplingParams

import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ScriptArguments:
    model_name: Optional[str] = None
    """Name of the model"""
    base_model_name: Optional[str] = None
    """Name of the base model"""
    output_filepath: Optional[str] = None
    """Where to save the generated descriptions"""
    eval_dataset: Optional[str] = None
    """Image dataset used for calculating the reward"""


args: ScriptArguments = tyro.cli(ScriptArguments)

assert args.output_filepath not in ["none", None]


print("Loading eval query dataset")
_, _, eval_queries = utils.single_class_text_dataset(args.eval_dataset)


tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    raise NotImplementedError

tmp_device_map = "cpu"

with SoftTemporaryDirectory() as tmp:
    model_ = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map=tmp_device_map,
    )
    model_ = PeftModel.from_pretrained(model_, args.model_name)
    model_ = model_.merge_and_unload()

    tmp_model_name = Path(tmp) / args.model_name
    model_.save_pretrained(tmp_model_name, max_shard_size="1GB")
    del model_

    gc.collect()
    if is_xpu_available():
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()

    model = LLM(
        model=tmp_model_name,
        tokenizer=args.model_name,
        trust_remote_code=True,
        revision=None,
        tokenizer_revision=None,
        max_model_len=2048,
    )

_gen_kwargs = dict()
_gen_kwargs["top_k"] = -1
sampling_params = SamplingParams(
    max_tokens=32,
    top_p=1.0,
    stop_token_ids=[tokenizer.eos_token_id],
    **_gen_kwargs,
)


all_eval_queries = list(itertools.chain(*eval_queries))
global_cls_query_sizes = [len(i) for i in eval_queries]

with torch.no_grad():
    outputs = model.generate(all_eval_queries, sampling_params)
    all_eval_descriptions = [output.outputs[0].text for output in outputs]
cum_sizes = [0] + list(itertools.accumulate(global_cls_query_sizes))
eval_descriptions = [
    all_eval_descriptions[cum_sizes[i - 1] : cum_sizes[i]]
    for i in range(1, len(cum_sizes))
]

filepath = Path(args.output_filepath)
filepath.parent.mkdir(exist_ok=True, parents=True)
with filepath.open("w") as f:
    json.dump(eval_descriptions, f)
