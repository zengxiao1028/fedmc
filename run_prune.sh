#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./federated"

python federated_trainer.py --task=emnist_cr \
                            --total_rounds=1500 \
                            --client_optimizer=adam \
                            --client_learning_rate=0.01 \
                            --client_batch_size=20 \
                            --server_optimizer=sgd \
                            --server_learning_rate=1.0 \
                            --clients_per_round=50 \
                            --client_epochs_per_round=1 \
                            --rounds_per_checkpoint=100 \
                            --experiment_name=emnist_fedavg_adam2 \
                            --enable_prune=false \
                            --root_output_dir=./result \

#sudo shutdown +1