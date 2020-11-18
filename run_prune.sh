#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./federated"

python federated_trainer.py --task=emnist_cr \
                            --total_rounds=1500 \
                            --client_optimizer=sgd \
                            --client_learning_rate=0.1 \
                            --client_batch_size=20 \
                            --server_optimizer=sgd \
                            --server_learning_rate=1.0 \
                            --clients_per_round=50 \
                            --client_epochs_per_round=1 \
                            --experiment_name=prune_emnist_fedavg \
                            --root_output_dir=./result