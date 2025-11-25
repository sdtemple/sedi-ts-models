#!/bin/bash

mkdir -p outputs

window_denom=3

function floor() {
    local num=$1
    # Use bc to perform the floor calculation
    # For positive numbers, this is simple truncation
    # For negative numbers, more logic is needed to go towards negative infinity
    echo "scale=0; ($num / 1)" | bc
}

# long short-term memory (LSTM) models
for window_size in 30 60 90
    do for lstm_hidden_size in 32 64
        do for linear_hidden_size in 32
            do for lstm_num_layers in 2
                do for lambda_underestimate in 1.2 1.5 1.8
                    do sbatch \
                        --account=sethtem0 \
                        --mem=48G \
                        --time=08:00:00 \
                        --gpus-per-node=1 \
                        --cpus-per-task=1 \
                        --partition=spgpu \
                        run-kfold.sh \ 
                        lstm_wsz${window_size}_wst$(floor $window_size/$window_denom)_lstmsize${lstm_hidden_size}_linearsize${linear_hidden_size}_lstmlayers${lstm_num_layers}_lambdaunder${lambda_underestimate}
                    done
                done
            done
        done
    done

# # gated recurrent unit (GRU) models
# for window_size in 30 60 90
#     do for lstm_hidden_size in 32 64
#         do for linear_hidden_size in 32
#             do for lstm_num_layers in 2
#                 do for lambda_underestimate in 1.2 1.5 1.8
#                     do sbatch \
#                         --account=sethtem0 \
#                         --mem=48G \
#                         --time=08:00:00 \
#                         --gpus-per-node=1 \
#                         --cpus-per-task=1 \
#                         --partition=spgpu \
#                         run-kfold.sh \ 
#                         gru_wsz${window_size}_wst$(floor $window_size/$window_denom)_lstmsize${lstm_hidden_size}_linearsize${linear_hidden_size}_lstmlayers${lstm_num_layers}_lambdaunder${lambda_underestimate}
#                     done
#                 done
#             done
#         done
#     done
