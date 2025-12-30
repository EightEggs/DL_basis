import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("phi-4-mini-instruct", "GPU")

for token in pipe.generate("How to make you an API instead of cli command", max_new_tokens=200, stream=True):
    print(token, end='', flush=True)