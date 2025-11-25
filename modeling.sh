#!/bin/bash

mkdir -p configs
rm -f configs/*.json

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
                    do python write-json-file.py \
                        --output-path configs/lstm_wsz${window_size}_wst$(floor $window_size/$window_denom)_lstmsize${lstm_hidden_size}_linearsize${linear_hidden_size}_lstmlayers${lstm_num_layers}_lambdaunder${lambda_underestimate}.json \
                        --window-size ${window_size} \
                        --window-step $(floor $window_size/$window_denom) \
                        --lstm-hidden-size ${lstm_hidden_size} \
                        --linear-hidden-size ${linear_hidden_size} \
                        --lambda-underestimate ${lambda_underestimate} \
                        --lstm-num-layers ${lstm_num_layers}
                    done
                done
            done
        done
    done

# # gated recurrent unit (GRU) models
# for window_size in 30 60 90
#     do for lstm_hidden_size in 32 64
#         do for linear_hidden_size in 32
#             do for lstm_num_layers in 1 2
#                 do for lambda_underestimate in 1.2 1.5 1.8
#                     do python write-json-file.py \
#                         --output-path "configs/gru_wsz${window_size}_wst$(floor $window_size/$window_denom)_lstmsize${lstm_hidden_size}_linearsize${linear_hidden_size}_lstmlayers${lstm_num_layers}_lambdaunder${lambda_underestimate}.json" \
#                         --window-size ${window_size} \
#                         --window-step $(floor $window_size/$window_denom) \
#                         --lstm-hidden-size ${lstm_hidden_size} \
#                         --linear-hidden-size ${linear_hidden_size} \
#                         --lambda-underestimate ${lambda_underestimate} \
#                         --lstm-num-layers ${lstm_num_layers} \
#                         --gru
#                     done
#                 done
#             done
#         done
#     done
