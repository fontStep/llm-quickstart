from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    print("json_post_list:",json_post_list)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer

def set_env():
    import subprocess
    import os
    os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface/cache'
    os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp/huggingface/cache'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
    output = result.stdout
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value

def load_model():
    from transformers import BitsAndBytesConfig
         #加载量化配置
    _compute_dtype_map = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16
    }

        # QLoRA 量化配置
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type='nf4',
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=_compute_dtype_map['bf16'])

        # 加载量化后模型(与微调的 revision 保持一致）
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b",
                                            quantization_config=q_config,
                                            device_map='auto',
                                            trust_remote_code=True,
                                            revision='b098244'
    
                                            )
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    
    return model,tokenizer
     
                    

if __name__ == "__main__":
    set_env()
    model ,tokenizer = load_model()
    # print("model:",model)
    # print("tokenizer:",tokenizer)
    # model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8081, workers=1)