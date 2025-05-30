# use python=3.11
# create a conda environment
# conda create -n pcd python=3.11
# conda activate pcd

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "jax[cuda12]==0.5.0"

cd packages/openpi-client
pip install -e .
cd ../..

cd lerobot-v2
pip install -e .
cd ..

# install grounded sam 2
cd src/openpi/models/contrast_utils/grounded_sam_2
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd ../../../../..

# install SED (optional)
# cd src/openpi/models/contrast_utils/SED/open_clip
# make install
# cd ../../../../../..
# pip install -r src/openpi/models/contrast_utils/SED/requirements.txt

# install inpaint-anything
pip install -r src/openpi/models/contrast_utils/inpaint_anything/lama/requirements.txt

pip install -e .

pip install numpy==1.26.4
pip install hydra-core==1.3.2