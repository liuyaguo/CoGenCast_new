<h1 align="center">CoGenCast: A Coupled Autoregressive-Flow Generative Framework<br>for Time Series Forecasting</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.2.2" />
  <img src="https://img.shields.io/badge/Time%20Series-Forecasting-2F4F4F" alt="Time Series Forecasting" />
</p>

## <img src="https://raw.githubusercontent.com/feathericons/feather/master/icons/file-text.svg" width="18" height="18" alt="Abstract icon" /> Abstract
We propose CoGenCast, a coupled autoregressive-flow generative framework for time series forecasting. The method integrates an autoregressive backbone with flow-based generation to model complex temporal dependencies, aiming to improve forecasting accuracy and robustness across diverse datasets.

## <img src="https://raw.githubusercontent.com/feathericons/feather/master/icons/image.svg" width="18" height="18" alt="Overview icon" /> Overview
![CoGenCast Framework](assets/framework.png)

## <img src="https://raw.githubusercontent.com/feathericons/feather/master/icons/star.svg" width="18" height="18" alt="Key Features icon" /> Key Features
- Coupled autoregressive-flow generation for expressive forecasting distributions
- Flexible backbone integration with large language models for sequence modeling
- Supports pretraining and finetuning across multiple time-series datasets

## <img src="https://raw.githubusercontent.com/feathericons/feather/master/icons/rocket.svg" width="18" height="18" alt="Quick Start icon" /> Quick Start
### Environment
- Python 3.10 (recommended)
- Install dependencies:

```bash
pip install -r requirements.txt
pip install transformers
```

### Datasets and Qwen3-0.6B
- Datasets: download the datasets you need and place them under `./datasets/` (e.g., `./datasets/ETTh1/ETTh1.csv`).
- Qwen3-0.6B: download the Qwen3-0.6B weights from [Hugging Face](https://huggingface.co/) and set the local path via `--llm_path`.

### Example Run
```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name finetune \
  --is_training 1 \
  --root_path ./datasets/Wind/ \
  --data_path Wind.csv \
  --model_id Wind \
  --model CoGenCast \
  --data Wind \
  --features M \
  --input_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --e_layers 2 \
  --pt_layers 4 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --n_heads 16 \
  --d_model 1024 \
  --d_ff 256 \
  --patch_len 4 \
  --stride 4 \
  --dropout 0.2 \
  --head_dropout 0.1 \
  --batch_size 4 \
  --gpu 0 \
  --lr_decay 0.5 \
  --lradj step \
  --time_steps 1000 \
  --scheduler cosine \
  --patience 3 \
  --backbone Qwen3-0.6B \
  --learning_rate 1e-4 \
  --pct_start 0.3
```

## <img src="https://raw.githubusercontent.com/feathericons/feather/master/icons/bar-chart-2.svg" width="18" height="18" alt="Performance icon" /> Performance
![Main Results](assets/table.png)

## <img src="https://raw.githubusercontent.com/feathericons/feather/master/icons/heart.svg" width="18" height="18" alt="Acknowledgement icon" /> Acknowledgement
We thank the open-source community for the datasets and foundational libraries used in this project.
