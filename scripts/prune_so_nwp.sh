#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./federated"

python federated_trainer.py --task=stackoverflow_nwp \
                            --so_nwp_vocab_size=10000 \
                            --so_nwp_num_oov_buckets=1 \
                            --so_nwp_sequence_length=20 \
                            --so_nwp_num_validation_examples=10000 \
                            --so_nwp_max_elements_per_user=1000 \
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
                            --clients_per_round=50 \
                            --client_epochs_per_round=1 \
                            --server_optimizer=adam \
                            --server_learning_rate=0.01 \
                            --server_adam_beta_1=0.9 \
                            --server_adam_beta_2=0.99 \
                            --server_adam_epsilon=0.0001 \
                            --rounds_per_checkpoint=100 \
                            --rounds_per_eval=10 \
                            --experiment_name=prune_so_fedavg3 \
                            --enable_prune=true \
                            --begin_step=300 \
                            --end_step=1500 \
                            --initial_sparsity=0.0 \
                            --final_sparsity=0.9 \
                            --root_output_dir=./result_so


#sudo shutdown +1