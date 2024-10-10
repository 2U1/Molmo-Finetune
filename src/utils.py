from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoProcessor, AutoConfig
import warnings
import os

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# This code is borrowed from LLaVA
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map}
    
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['_attn_implementation'] = 'flash_attention_2'

    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        # lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        # if hasattr(lora_cfg_pretrained, 'quantization_config'):
        #     del lora_cfg_pretrained.quantization_config
        processor = AutoProcessor.from_pretrained(model_base, trust_remote_code=True)
        print('Loading Molmo from base model...')
        # model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, trust_remote_code=True, **kwargs)

        # Using the default config from the huggingface model
        # Looks like the model doesn't accpet the config local config argument for now.
        model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
        token_num, tokem_dim = model.model.transformer.ff_out.out_features, model.model.transformer.ff_out.in_features
        if model.model.transformer.ff_out.weight.shape[0] != token_num:
            print('Resizing the model to match the Molmo token size...')
            model.model.transformer.ff_out.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.transformer.wte.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Molmo weights...')
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)

        print('Merging LoRA weights...')
        model = model.merge_and_unload()

        print('Model Loaded!!!')

    else:
        processor = AutoProcessor.from_pretrained(model_base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)

    return processor, model


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]