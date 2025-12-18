# s2l-rl
s2l for reinforcement learning

## Set Up
```sh 
git clone git@github.com:giyushino/s2l-rl.git
cd s2l-rl
conda create -n s2l python==3.12
conda activate s2l
cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install -e . --no-deps
cd ..
```
