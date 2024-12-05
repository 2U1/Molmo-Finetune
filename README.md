# Molmo-Finetune

This repository contains a script for training [Molmo Series](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19) with using HuggingFace.

However the model uploaded at the huggingfece hub is a sort of a preview version that has few limitations.

- **Only supports fp32 (Can supprot fp16 or bf16 however not stable)**
- **Only single-image is supported**
- **Grad Checkpointing disabled**
- **Flash-attention and sdpa disabled**

**Note:** This repo isn't maintained until the official version (Not the preview one) is released.

## Other projects

**[[Phi3-Vision Finetuning]](https://github.com/2U1/Phi3-Vision-Finetune)**<br>
**[[Qwen2-VL Finetuning]](https://github.com/2U1/Qwen2-VL-Finetune)**<br>
**[[LLAMA3.2-Vision Finetuning]](https://github.com/2U1/Llama3.2-Vision-Finetune)**<br>
**[[Pixtral Finetune]](https://github.com/2U1/Pixtral-Finetune)**

## Update

- [2024/12/04] Add memory efficient 8-bit training.

## Table of Contents

- [Molmo-Finetune](#molmo-finetune)
  - [Other projects](#other-projects)
  - [Table of Contents](#table-of-contents)
  - [Supported Features](#supported-features)
  - [Installation](#installation)
    - [Using `environment.yaml`](#using-environmentyaml)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
    - [Full Finetuning](#full-finetuning)
    - [Finetune with LoRA](#finetune-with-lora)
      - [Merge LoRA Weights](#merge-lora-weights)
      - [Issue for libcudnn error](#issue-for-libcudnn-error)
  - [TODO](#todo)
  - [Known Issues](#known-issues)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Supported Features

- Deepspeed
- LoRA, QLoRA
- Full-finetuning

## Installation

Install the required packages using `environment.yaml`.

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate molmo
```

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**
**Please see the example below and follow format your data.**

<details>
<summary>Example for single image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

## Training

**Note:** The model was updated to use bf16 or fp16 however, the output could be chagned compared to fp32.

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

**IMPORTANT:** As the model is a preview version, it is a bit unstable when using LoRA. It is preferred to use full finetuning for now.

If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/finetune_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

~~**IMPORTANT:** If you want to tune the `wte` with LoRA, You need to tune `ff_out` (that is `lm_head` in other models) together.~~
**NOTE:** I couldn't exactly find the `embedding layer` that has weight so, the `ff_out` layer should be fine tuned temporarily.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Llama3.2-Vision model. **(Required)**
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--tune_projector` (bool): Option to tune projector (default: True).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--projector_lr` (float): Learning rate for projector.
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 128K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

If you run out of vram, you can use [zero3_offload](./scripts/zero3_offload.json) instead of [zero3](./scripts/zero3_offload.json). However, using zero3 is preferred.

#### Merge LoRA Weights

```
bash scripts/merge_lora.sh
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths. (Also in `merge_lora.sh` when using LoRA.)

#### Issue for libcudnn error

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

You could run `unset LD_LIBRARY_PATH` for this error.
You could see this [issue](https://github.com/andimarafioti/florence2-finetuning/issues/2)

## TODO

- [ ] Update easy use of adam-8bit

## Known Issues

- [libcudnn issue](#issue-for-libcudnn-error)

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{Molmo-Finetuning,
  author = {Yuwon Lee},
  title = {Molmo-Finetune},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Molmo-Finetune}
}
```

## Acknowledgement

This project is based on

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.
- [Molmo Series](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19): Awesome pretrained MLLM by AllenAI.
