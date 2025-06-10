#!/bin/bash

# Configurable variables
TMUX_SESSION="server"
ENV_NAME="myenv1"
CONTAINER_NAME="my_container"
PYTHON_SCRIPT_PATH="/app/script.py"   # Path inside the container
START_CONTAINER_SCRIPT="sudo ./hailo_ai_sw_suite_docker_run.sh --resume"
SUDO_PASSWORD="amax1234"  # ⚠️ Not secure
PI_USER="amax"
PI_HOST="raspberrypi.local"
PI_SESSION="pi_server"
PI_REMOTE_DIR="/home/amax/GitHub/newbranch/HI4Lines_Insp/deploy/rpi/deploy/"
PI_PYTHON_SCRIPT="run_classifier_evaluator_server.py"                # Name of the .py file on the Pi


# Create a new detached tmux session
tmux new-session -d -s "$TMUX_SESSION"
tmux split-window -v -t "$TMUX_SESSION"

tmux send-keys -t "$TMUX_SESSION:0.0" "conda activate $ENV_NAME && cd /home/amax/machairas/HI4Lines_Insp/hi4lines_insp/" C-m

# Run the start_container.sh script in tmux
tmux send-keys -t "$TMUX_SESSION:0.1" "cd /home/amax/machairas" C-m
tmux send-keys -t "$TMUX_SESSION:0.1" "echo '$SUDO_PASSWORD' | sudo -S $START_CONTAINER_SCRIPT" C-m
tmux send-keys -t "$TMUX_SESSION:0.1" "$START_CONTAINER_SCRIPT" C-m

# Wait a few seconds to ensure the container starts
sleep 3

# Run the Python script inside the Docker container
tmux pipe-pane -t "$TMUX_SESSION:0.1" "grep --line-buffered 'Running' >> /home/amax/machairas/a.txt"
echo a.txt
tmux send-keys -t "$TMUX_SESSION:0.1" "cd ../home/amax/HI4Lines_Insp/hailo_src/ && python3 server.py" C-m

tmux split-window -h -t "$TMUX_SESSION:0.1"

tmux send-keys -t "$TMUX_SESSION:0.2" \
  "ssh -t $PI_USER@$PI_HOST '\
    # create a detached tmux session if it doesn't exist;\
    tmux new-session -d -s $PI_SESSION;\
    # send commands into that session;\
    tmux send-keys -t $PI_SESSION \"conda activate HI4Lines_Insp && cd $PI_REMOTE_DIR && python3 $PI_PYTHON_SCRIPT\" C-m;\
    # (optional) attach to it so you can watch it live;\
    tmux attach -t $PI_SESSION\
  '" C-m

tmux attach -t "$TMUX_SESSIONN"
