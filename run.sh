#!/bin/bash

# Usage: ./run.sh [train|inference] [config_file]
# Example: ./run.sh train conf/cifar10.json
# Example: ./run.sh inference conf/cifar100.json

MODE=$1
CONFIG=$2

if [ -z "$MODE" ]; then
    echo "Usage: ./run.sh [train|inference] [config_file]"
    exit 1
fi

if [ -z "$CONFIG" ]; then
    CONFIG="conf/cifar10.json"
    echo "No config specified, using default: $CONFIG"
fi

if [ "$MODE" == "train" ]; then
    echo "Starting Training with config: $CONFIG"
    python -m src.train_backbone --config "$CONFIG"
elif [ "$MODE" == "inference" ]; then
    echo "Starting Inference with config: $CONFIG"
    python -m src.inference --config "$CONFIG"
else
    echo "Invalid mode: $MODE. Use 'train' or 'inference'."
    exit 1
fi
