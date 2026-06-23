"""Inferência rápida com ONNX Runtime (3-5x mais rápido que PyTorch em CPU)."""
import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial import distance
from itertools import groupby
from tqdm import tqdm
import argparse
import time


def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps


def postprocess(output, width=640, height=360):
    """Extrai (x, y) do heatmap de saída do modelo."""
    output = output.squeeze()
    # output shape: (H, W) — índice com valor máximo é a posição da bola
    idx = np.argmax(output)
    if output.flat[idx] == 0:
        return None, None
    y = idx // width
    x = idx % width
    return int(x), int(y)


def infer_model_onnx(frames, session):
    height, width = 360, 640
    dists = [-1] * 2
    ball_track = [(None, None)] * 2
    t0 = time.time()

    for num in tqdm(range(2, len(frames))):
        img       = cv2.resize(frames[num],   (width, height))
        img_prev  = cv2.resize(frames[num-1], (width, height))
        img_pre2  = cv2.resize(frames[num-2], (width, height))
        imgs = np.concatenate((img, img_prev, img_pre2), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = session.run(None, {'input': inp})[0]
        x_pred, y_pred = postprocess(out)
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)

    elapsed = time.time() - t0
    fps_proc = len(frames) / elapsed
    print(f"Tempo de processamento: {elapsed:.1f}s ({fps_proc:.2f} frames/s)")
    return ball_track, dists


def remove_outliers(ball_track, dists, max_dist=100):
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if i + 1 < len(dists) and ((dists[i+1] > max_dist) or (dists[i+1] == -1)):
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif i > 0 and dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    return ball_track


def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]
    cursor, min_value, result = 0, 0, []
    for i, (k, l) in enumerate(groups):
        if (k == 1) and (i > 0) and (i < len(groups) - 1):
            if ball_track[cursor-1][0] and ball_track[cursor+l][0]:
                dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
                if (l >= max_gap) or (dist / l > max_dist_gap):
                    if cursor - min_value > min_track:
                        result.append([min_value, cursor])
                        min_value = cursor + l - 1
        cursor += l
    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])
    return result


def interpolation(coords):
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    x = np.array([p[0] if p[0] is not None else np.nan for p in coords])
    y = np.array([p[1] if p[1] is not None else np.nan for p in coords])
    nons, yy = nan_helper(x)
    x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])
    return list(zip(x, y))


def write_track(frames, ball_track, path_output, fps, trace=15):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    detected = sum(1 for p in ball_track if p[0] is not None)
    print(f"Frames com bola detectada: {detected}/{len(ball_track)} ({100*detected//len(ball_track)}%)")
    for num in range(len(frames)):
        frame = frames[num].copy()
        for i in range(trace):
            if (num - i) >= 0 and ball_track[num-i][0] is not None:
                x = int(ball_track[num-i][0])
                y = int(ball_track[num-i][1])
                alpha = (trace - i) / trace
                radius = max(2, int(8 * alpha))
                thickness = max(1, int(10 * alpha))
                color = (0, int(255 * alpha), int(255 * alpha))
                cv2.circle(frame, (x, y), radius, color, thickness)
            elif (num - i) >= 0 and ball_track[num-i][0] is None:
                break
        out.write(frame)
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default='weights/tracknet.onnx')
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--video_out_path', type=str, default='output.mp4')
    parser.add_argument('--extrapolation', action='store_true')
    parser.add_argument('--trace', type=int, default=15)
    args = parser.parse_args()

    session = ort.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])
    print(f"Modelo ONNX carregado: {args.onnx_path}")

    print(f"Lendo vídeo: {args.video_path}")
    frames, fps = read_video(args.video_path)
    print(f"Total de frames: {len(frames)}, FPS: {fps}")

    ball_track, dists = infer_model_onnx(frames, session)
    ball_track = remove_outliers(ball_track, dists)

    if args.extrapolation:
        subtracks = split_track(ball_track)
        for r in subtracks:
            sub = interpolation(ball_track[r[0]:r[1]])
            ball_track[r[0]:r[1]] = sub

    write_track(frames, ball_track, args.video_out_path, fps, trace=args.trace)
    print(f"Vídeo salvo em: {args.video_out_path}")
