#!/bin/bash


export PYTHONPATH="${PYTHONPATH}:./federated"


python federated/optimization/main/federated_trainer.py \
                            --task=stackoverflow_nwp \
                            --so_nwp_vocab_size=10000 \
                            --so_nwp_num_oov_buckets=1 \
                            --so_nwp_sequence_length=20 \
                            --so_nwp_num_validation_examples=10000 \
                            --so_nwp_max_elements_per_user=128 \
                            --so_nwp_embedding_size=96 \
                            --so_nwp_latent_size=670 \
                            --so_nwp_num_layers=1 \
                            --total_rounds=1500 \
                            --client_optimizer=sgd \
                            --client_learning_rate=0.316 \
                            --client_lr_schedule=exp_decay \
                            --client_lr_decay_steps=500 \
                            --client_lr_decay_rate=0.1 \
                            --client_lr_staircase=true \
                            --client_batch_size=16 \
                            --clients_per_round=10 \
                            --client_epochs_per_round=1 \
                            --server_optimizer=adam \
                            --server_learning_rate=0.01 \
                            --server_adam_beta_1=0.9 \
                            --server_adam_beta_2=0.99 \
                            --server_adam_epsilon=0.0001 \
                            --rounds_per_checkpoint=100 \
                            --experiment_name=so_fedavg_official2 \
                            --root_output_dir=./result_so