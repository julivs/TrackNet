"""Exporta o modelo TrackNet para ONNX para inferência mais rápida em CPU."""
from model import BallTrackerNet
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to .pt weights')
    parser.add_argument('--onnx_path', type=str, default='weights/tracknet.onnx', help='path to save .onnx')
    args = parser.parse_args()

    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu', weights_only=False))
    model.eval()

    # Input: 1 sample, 9 canais (3 frames RGB), 360x640
    dummy = torch.zeros(1, 9, 360, 640)

    torch.onnx.export(
        model, dummy, args.onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11,
    )
    print(f"Modelo exportado para: {args.onnx_path}")
