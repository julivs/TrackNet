from model import BallTrackerNet
import torch
import cv2
from general import postprocess
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance

def read_video(path_video):
    """ Read video file    
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
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

def infer_model(frames, model, device):
    """ Run pretrained model on a consecutive list of frames
    :params
        frames: list of consecutive video frames
        model: pretrained model
        device: torch device ('cpu' or 'cuda')
    :return
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """
    infer_h, infer_w = 360, 640
    orig_h, orig_w = frames[0].shape[:2]
    scale_x = orig_w / infer_w
    scale_y = orig_h / infer_h
    dists = [-1]*2
    ball_track = [(None,None)]*2
    for num in tqdm(range(2, len(frames))):
        img = cv2.resize(frames[num], (infer_w, infer_h))
        img_prev = cv2.resize(frames[num-1], (infer_w, infer_h))
        img_preprev = cv2.resize(frames[num-2], (infer_w, infer_h))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)
        if x_pred is not None:
            x_pred = int(x_pred * scale_x)
            y_pred = int(y_pred * scale_y)
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)
    return ball_track, dists

def remove_outliers(ball_track, dists, max_dist = 100):
    """ Remove outliers from model prediction    
    :params
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
        max_dist: maximum distance between two neighbouring ball points
    :return
        ball_track: list of ball points
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    return ball_track  

def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """ Split ball track into several subtracks in each of which we will perform
    ball interpolation.    
    :params
        ball_track: list of detected ball points
        max_gap: maximun number of coherent None values for interpolation  
        max_dist_gap: maximum distance at which neighboring points remain in one subtrack
        min_track: minimum number of frames in each subtrack    
    :return
        result: list of subtrack indexes    
    """
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
            if (l >=max_gap) | (dist/l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1        
        cursor += l
    if len(list_det) - min_value > min_track: 
        result.append([min_value, len(list_det)]) 
    return result    

def interpolation(coords):
    """ Run ball interpolation in one subtrack    
    :params
        coords: list of ball coordinates of one subtrack    
    :return
        track: list of interpolated ball coordinates of one subtrack
    """
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans]= np.interp(xx(nans), xx(~nans), y[~nans])

    track = [*zip(x,y)]
    return track

def write_track(frames, ball_track, path_output_video, fps, trace=15):
    """ Write video file with detected ball tracks and trail effect
    :params
        frames: list of original video frames
        ball_track: list of ball coordinates
        path_output_video: path to output video
        fps: frames per second
        trace: number of frames in the trail
    """
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_output_video, fourcc, fps, (width, height))
    detected = sum(1 for p in ball_track if p[0] is not None)
    print(f"Frames com bola detectada: {detected}/{len(ball_track)} ({100*detected//len(ball_track)}%)")
    for num in range(len(frames)):
        frame = frames[num].copy()
        for i in range(trace):
            if (num - i) >= 0:
                if ball_track[num-i][0] is not None:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    alpha = (trace - i) / trace
                    radius = max(2, int(8 * alpha))
                    thickness = max(1, int(10 * alpha))
                    color = (0, int(255 * alpha), int(255 * alpha))
                    cv2.circle(frame, (x, y), radius, color, thickness)
                else:
                    break
        out.write(frame)
    out.release()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--model_path', type=str, help='path to model weights')
    parser.add_argument('--video_path', type=str, help='path to input video')
    parser.add_argument('--video_out_path', type=str, default='output.mp4', help='path to output video (.mp4)')
    parser.add_argument('--extrapolation', action='store_true', help='interpolate missing ball positions')
    parser.add_argument('--trace', type=int, default=15, help='number of frames in trail (default: 15)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando device: {device}")

    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()

    print(f"Lendo vídeo: {args.video_path}")
    frames, fps = read_video(args.video_path)
    print(f"Total de frames: {len(frames)}, FPS: {fps}")

    ball_track, dists = infer_model(frames, model, device)
    ball_track = remove_outliers(ball_track, dists)

    if args.extrapolation:
        subtracks = split_track(ball_track)
        for r in subtracks:
            ball_subtrack = ball_track[r[0]:r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_track[r[0]:r[1]] = ball_subtrack

    write_track(frames, ball_track, args.video_out_path, fps, trace=args.trace)
    print(f"Vídeo salvo em: {args.video_out_path}")
    
    
    
    
    
