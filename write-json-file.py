import argparse
import json
from pathlib import Path

DEFAULT = {
    "window_size": 30,
    "window_step": 10,
    "num_stations": 290,
    "X_scaler_path": "X_scaler.joblib",
    "Y_scaler_path": "Y_scaler.joblib",
    "lstm_hidden_size": 64,
    "linear_hidden_size": 64,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.2,
    "linear_dropout": 0.5,
    "if_layer_norm": False,
    "if_gru": False,
    "batch_size": 32,
    "num_epochs": 200,
    "learning_rate": 0.001,
    "up_th": 0.9,
    "down_th": 0.1,
    "lambda_underestimate": 1.5,
    "lambda_overestimate": 1.0,
    "lambda_init": 1.0
}

def parse_args():
    p = argparse.ArgumentParser(description="Create or update an LSTM JSON config.")
    p.add_argument("--output-path", dest="output_path", default="lstm.json", help="Output JSON path (default: lstm.json).")
    p.add_argument("--lstm-hidden-size", type=int, help="lstm_hidden_size", default=None)
    p.add_argument("--linear-hidden-size", type=int, help="linear_hidden_size", default=None)
    p.add_argument("--lstm-num-layers", type=int, help="lstm_num_layers", default=None)
    p.add_argument("--window-step", type=int, help="window_step", default=None)
    p.add_argument("--window-size", type=int, help="window_size", default=None)
    p.add_argument("--X-scaler-path", dest="X_scaler_path", help="X_scaler_path", default=None)
    p.add_argument("--Y-scaler-path", dest="Y_scaler_path", help="Y_scaler_path", default=None)
    p.add_argument("--lambda-underestimate", type=float, help="lambda_underestimate", default=None)
    p.add_argument("--up-th", type=float, help="up_th", default=None)
    p.add_argument("--gru", dest="if_gru", action="store_true", help="enable if_gru")
    p.add_argument("--layer-norm", dest="if_layer_norm", action="store_true", help="enable if_layer_norm")
    p.set_defaults(if_gru=None, if_layer_norm=None)
    return p.parse_args()

def main():
    args = parse_args()
    config = DEFAULT.copy()

    # Apply overrides if provided
    if args.lstm_hidden_size is not None:
        config["lstm_hidden_size"] = args.lstm_hidden_size
    if args.linear_hidden_size is not None:
        config["linear_hidden_size"] = args.linear_hidden_size
    if args.lstm_num_layers is not None:
        config["lstm_num_layers"] = args.lstm_num_layers
    if args.window_size is not None:
        config["window_size"] = args.window_size
    if args.window_step is not None:
        config['window_step'] = args.window_step
    if args.X_scaler_path is not None:
        config["X_scaler_path"] = args.X_scaler_path
    if args.Y_scaler_path is not None:
        config["Y_scaler_path"] = args.Y_scaler_path
    if args.lambda_underestimate is not None:
        config["lambda_underestimate"] = args.lambda_underestimate
    if args.up_th is not None:
        config["up_th"] = args.up_th
        # symmetry
        config['down_th'] = 1.0 - args.up_th
    if args.if_gru is not None:
        config["if_gru"] = args.if_gru
    if args.if_layer_norm is not None:
        config["if_layer_norm"] = args.if_layer_norm

    out_path = Path(args.output_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, sort_keys=False)
    print(f"Wrote config to: {out_path.resolve()}")

if __name__ == "__main__":
    main()