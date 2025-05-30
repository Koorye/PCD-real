# Policy Contrastive Decoding for Robotic Foundation Models (Real-world)

> **Note**: This is the real-world version of our work. If you are looking for the SIMPLER version, please check out [PCD](https://github.com/Koorye/PCD).

Official implementation of the paper "[Policy Contrastive Decoding for Robotic Foundation Models](https://arxiv.org/abs/2505.13255)".

> **Note**: We are doing our best to improve this work. If you have any questions or suggestions, please feel free to create an issue in this repo or contact us at shihan.wu.koorye@outlook.com.

[[Project]](https://koorye.github.io/proj/PCD) [[ArXiv]](https://arxiv.org/abs/2505.13255) [[PDF]](https://arxiv.org/pdf/2505.13255) [[PCD]](https://github.com/Koorye/PCD)

## News
- 🔥**May 23, 2025**: Our paper has been updated for better clarity and readability. The optimized version is now available on arXiv.
- 🔥**May 20, 2025**: The code is released and the paper is now available on arXiv.

## Running 

1. Clone this repository:

```bash
git clone https://github.com/Koorye/PCD-real.git
```

2. Install the required packages:

```bash
bash scripts/install_dependencies.sh
```

3. Download the pretrained checkpoints:

> **Note**: Some of the checkpoints cannot be downloaded directly, you may need to download them manually from the links provided in the script.

```bash
bash scripts/download_pretrained_weights.sh
```

4. Fine-tune your own model. More details can be found in official repository of [Pi-0](https://github.com/Physical-Intelligence/openpi).

5. Deploy the model:

**Baseline**

```bash
python scripts/deploy_server.py \
    --config_name your_config_name \ # example: pi0_real
    --checkpoint_bucket /path/to/your/finetuned/checkpoint/bucket
```

**+PCD (Ours)**

```bash
python scripts/deploy_server.py \
    --config_name your_config_name_pcd \ # example: pi0_real_pcd
    --checkpoint_bucket /path/to/your/finetuned/checkpoint/bucket
```

6. Implement your own inference script based on the provided example in `scripts/inference_test.py`.