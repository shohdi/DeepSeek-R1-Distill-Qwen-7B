# DeepSeek-R1-Distill-Qwen-7B
Run deepseek local on windows 11 , pytorch and rtx 4060 laptop with 32 gb ram <br/>
model repo :<br/>
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B<br/>
libraries to install :<br/>
pytroch with cuda from pytorch web site<br/>
conda install -c conda-forge flask-restful<br/>
pip install transformers sentencepiece accelerate<br/>
pip install bitsandbytes<br/>
<br/>
The model downloaded using these commands :<br/>
pip install huggingface_hub<br/>
huggingface-cli login<br/>
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --cache-dir ./models<br/>
<br/>
after download , model will be in snapshot folder<br/>
change the model path in the code and run runDeep.py<br/>
<br/>
Thanks<br/>




