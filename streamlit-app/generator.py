import torch
from transformers import AutoTokenizer
from transformers import MistralForCausalLM 

#Quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

def load_models(model_path = 'mistral_local'):
    model = MistralForCausalLM.from_pretrained(model_path, 
                                               local_files_only=True,
                                               quantization_config=bnb_config,
                                               device_map={"":0})
    

    tokenizer = AutoTokenizer.from_pretrained('mistral_local', local_files_only = True)
    #Addition of the Pad Token as the Tokenizer.pad_token is Set to None, this is required to ensure efficient padding
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token

    return model, tokenizer

def run_model(model, tokenizer, prompt = None): 
    #prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token        
    generate_ids = model.generate(inputs.input_ids, max_length=1000)
    resp = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    resp = resp.split('Answer:')

    return resp[0], resp[1]


# if __name__=="__main__":
#     model, tokenizer = load_models(model_path = "mistral_local")
#     resp = run_model(model, tokenizer, prompt)
