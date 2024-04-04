# Extract and Explore

Code for [If CLIP Could Talk: Understanding Vision-Language Model Representations Through Their Preferred Concept Descriptions](https://arxiv.org/abs/2403.16442)

> Recent works often assume that Vision-Language Model (VLM) representations are based on visual attributes like shape. However, it is unclear to what extent VLMs prioritize this information to represent concepts. We propose Extract and Explore (EX2), a novel approach to characterize important textual features for VLMs. EX2 uses reinforcement learning to align a large language model with VLM preferences and generates descriptions that incorporate the important features for the VLM. Then, we inspect the descriptions to identify the features that contribute to VLM representations. We find that spurious descriptions have a major role in VLM representations despite providing no helpful information, e.g., Click to enlarge photo of CONCEPT. More importantly, among informative descriptions, VLMs rely significantly on non-visual attributes like habitat to represent visual concepts. Also, our analysis reveals that different VLMs prioritize different attributes in their representations. Overall, we show that VLMs do not simply match images to scene descriptions and that non-visual or even spurious descriptions significantly influence their representations.

<img src="block_diagram.png" width=750>

## Setup

Install the following requirements:

```txt
accelerate==0.25.0
bitsandbytes==0.41.2.post2
datasets==2.15.0
open-clip-torch==2.23.0
peft==0.6.3.dev0
transformers==4.36.2
trl==0.7.5.dev0
vllm==0.2.7
torch==2.1.2
```

Follow the instructions in [this repo](https://github.com/BatsResearch/fudd/blob/main/Dataset_preparation.md) to prepare the datasets. Set the `DATA_ROOT` command line argument accordingly.

## Instructions

To analyze the representations of a contrastive VLM, first, use `train_runner.sh` to fine-tune and align an LLM with VLM preferences.
So, the LLM learns to generate descriptions that are closer to the corresponding images in the VLM embedding space.

After training, run `inference_runner.sh` to generate 25 descriptions that the VLM prioritizes for each concept.

Now, you can examine these descriptions to understand how the VLM represents each concept. For example, you can use `inspection_runner.sh` to ask ChatGPT if each description provides additional information about the corresponding concept.

To use [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) or [ALIGN](https://huggingface.co/docs/transformers/en/model_doc/align) checkpoints, just pass the model name on the Huggingface hub as the `vlm_name` argument. To use [OpenCLIP](https://github.com/mlfoundations/open_clip) models, set the `vlm_name` argument to `r-open-clip:MODEL:DATASET`, where `MODEL` is one of the models supported by OpenCLIP and `DATASET` is the pre-training dataset (e.g., `r-open-clip:ViT-bigG-14-CLIPA-336:datacomp1b`). To use OpenCLIP with Huggingface hub checkpoints, just use `r-open-clip:hf-hub:HUGGINGFACE_MODEL_NAME`, e.g., `r-open-clip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-384`.
