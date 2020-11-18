#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./federated"

python federated_trainer.py --task=emnist_cr \
                            --total_rounds=100 \
                            --client_optimizer=sgd \
                            --client_learning_rate=0.1 \
                            --client_batch_size=20 \
                            --server_optimizer=sgd \
                            --server_learning_rate=1.0 \
                            --clients_per_round=3 \
                            --client_epochs_per_round=1 \
                            --experiment_name=emnist_fedavg_experiment \
                            --init_checkpoint_dir=/Users/xiaozeng/research/FederatedLearning/fedmc/ckpt_0