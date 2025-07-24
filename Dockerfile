FROM jupyter/scipy-notebook

RUN pip install torch==2.2.2+cu121 torchaudio librosa matplotlib numpy \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple
