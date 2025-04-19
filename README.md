# DeepSeek-R1-Distill-Qwen-7B
Run deepseek local on windows 11 , pytorch and rtx 4060 laptop with 32 gb ram <br/>
model repo :
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
libraries to install :
pytroch with cuda from pytorch web site
pip install transformers sentencepiece accelerate
pip install bitsandbytes

The model downloaded using these commands :
pip install huggingface_hub
huggingface-cli login
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --cache-dir ./models

after download , model will be in snapshot folder
change the model path in the code and run runDeep.py

Thanks




