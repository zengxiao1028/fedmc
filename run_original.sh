#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./federated"


python federated/optimization/main/federated_trainer.py \
                            --task=emnist_cr \
                            --total_rounds=1500 \
                            --client_optimizer=sgd \
                            --client_learning_rate=0.0032 \
                            --client_lr_schedule=exp_decay \
                            --client_lr_decay_steps=500 \
                            --client_lr_decay_rate=0.1 \
                            --client_lr_staircase=true \
                            --client_batch_size=20 \
                            --clients_per_round=10 \
                            --client_epochs_per_round=1 \
                            --server_optimizer=adam \
                            --server_learning_rate=0.01 \
                            --server_adam_beta_1=0.9 \
                            --server_adam_beta_2=0.99 \
                            --experiment_name=emnist_fedavg_official_hp5 \
                            --root_output_dir=./result