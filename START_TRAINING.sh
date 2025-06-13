#!/bin/bash

# Configurable variables
TMUX_SESSION="server"
ENV_NAME="myenv1"
CONTAINER_NAME="my_container"
PYTHON_SCRIPT_PATH="/app/script.py"   # Path inside the container
START_CONTAINER_SCRIPT="sudo ./hailo_ai_sw_suite_docker_run.sh --resume"
SUDO_PASSWORD="amax1234"  # Not secure

PI_USER="amax"
PI_HOST="192.168.2.13"
PI_SESSION="server_rpi"
PI_REMOTE_DIR="/home/amax/GitHub/newbranch/HI4Lines_Insp/deploy/rpi/deploy/"
PI_PYTHON_SCRIPT="run_classifier_evaluator_server.py"                # Name of the .py file on the Pi
SUDO_PASSWORD_RPI="amax1234"

# Create a new detached tmux session
tmux new-session -d -s "$TMUX_SESSION"
tmux split-window -v -t "$TMUX_SESSION"

tmux send-keys -t "$TMUX_SESSION:0.0" "conda activate $ENV_NAME && cd /home/amax/HI4Lines_Insp/hi4lines_insp/ && python3 main_base_oneoff.py" C-m

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

#tmux send-keys -t "$TMUX_SESSION:0.2" "echo '$SUDO_PASSWORD_RPI' | ssh -t $PI_USER@$PI_HOST" C-m

tmux attach -t "$TMUX_SESSIONN"
