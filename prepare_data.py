# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

import h5py
from tqdm import tqdm

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="co-tracker/checkpoints/scaled_offline.pth",
        help="CoTracker model parameters",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="",
    )
    parser.add_argument("--grid_size", type=int, default=16, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
    )
    parser.set_defaults(offline=True)

    args = parser.parse_args()

    file_path = args.file_path

    with h5py.File(file_path, 'a') as f:
        f = f['data']
        # demos = list(f.keys())
        demos = ["demo_{}".format(i) for i in range(200, 1000)]
        print(f"Found demos: {demos}")

        for demo_name in tqdm(demos):
            print(f"\nProcessing demo: {demo_name}")

            demo_group = f[demo_name]
            if 'obs' in demo_group and 'agentview_image' in demo_group['obs']:
                agentview_image = demo_group['obs']['agentview_image']
                print(f"Shape of agentview_image: {agentview_image.shape}")
                
                video = torch.from_numpy(agentview_image[()]).permute(0, 3, 1, 2)[None].float()
                if args.checkpoint is not None:
                    if args.use_v2_model:
                        model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
                    else:
                        if args.offline:
                            window_len = 60
                        else:
                            window_len = 16
                        model = CoTrackerPredictor(
                            checkpoint=args.checkpoint,
                            v2=args.use_v2_model,
                            offline=args.offline,
                            window_len=window_len,
                        )
                else:
                    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

                model = model.to(DEFAULT_DEVICE)
                video = video.to(DEFAULT_DEVICE)

                pred_tracks, pred_visibility = model(
                    video,
                    grid_size=args.grid_size,
                    grid_query_frame=args.grid_query_frame,
                    backward_tracking=args.backward_tracking,
                )
                pred_tracks_np = pred_tracks.cpu().numpy()
                pred_visibility_np = pred_visibility.cpu().numpy()
                
                if 'tracks' in demo_group:
                    del demo_group['tracks']
                if 'visibility' in demo_group:
                    del demo_group['visibility']
                demo_group.create_dataset('tracks', data=pred_tracks_np[0] / 84.0)
                demo_group.create_dataset('visibility', data=pred_visibility_np)
                print("computed")

                vis = Visualizer(save_dir="./output_videos", pad_value=16, linewidth=1)
                vis.visualize(
                    video,
                    pred_tracks,
                    pred_visibility,
                    query_frame=0 if args.backward_tracking else args.grid_query_frame,
                    filename=f"{demo_name}",
                )
