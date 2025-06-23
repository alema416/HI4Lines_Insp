FROM continuumio/miniconda3:latest

# 1. Create the exact environment
COPY env.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean --all --yes \
    && rm /tmp/environment.yml

# 2. Initialize conda for bash and auto-activate myenv1 on login shells
RUN conda init bash \
    && echo "conda activate myenv1" >> /root/.bashrc

# 3. Put your code in place and install editable
WORKDIR /app

# 4. Make sure PATH picks up that env’s binaries if any notion of direct exec happens
ENV PATH=/opt/conda/envs/myenv1/bin:$PATH

# 5. Default to an interactive login shell, which will source ~/.bashrc
CMD ["bash", "-l"]
