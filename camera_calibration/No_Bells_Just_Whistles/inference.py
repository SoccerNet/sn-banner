import os
import cv2
import yaml
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as f

from tqdm import tqdm
from PIL import Image
from matplotlib.patches import Polygon

from .model.cls_hrnet import get_cls_net
from .model.cls_hrnet_l import get_cls_net as get_cls_net_l

# use utils.utils_calib2, a slight upgrade of utils.utils_calib
from .utils.utils_calib2 import FramebyFrameCalib, pan_tilt_roll_to_orientation
from .utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict,
)

transform2 = T.Resize((540, 960))

lines_coords = [
    [[0.0, 54.16, 0.0], [16.5, 54.16, 0.0]],
    [[16.5, 13.84, 0.0], [16.5, 54.16, 0.0]],
    [[16.5, 13.84, 0.0], [0.0, 13.84, 0.0]],
    [[88.5, 54.16, 0.0], [105.0, 54.16, 0.0]],
    [[88.5, 13.84, 0.0], [88.5, 54.16, 0.0]],
    [[88.5, 13.84, 0.0], [105.0, 13.84, 0.0]],
    [[0.0, 37.66, -2.44], [0.0, 30.34, -2.44]],
    [[0.0, 37.66, 0.0], [0.0, 37.66, -2.44]],
    [[0.0, 30.34, 0.0], [0.0, 30.34, -2.44]],
    [[105.0, 37.66, -2.44], [105.0, 30.34, -2.44]],
    [[105.0, 30.34, 0.0], [105.0, 30.34, -2.44]],
    [[105.0, 37.66, 0.0], [105.0, 37.66, -2.44]],
    [[52.5, 0.0, 0.0], [52.5, 68, 0.0]],
    [[0.0, 68.0, 0.0], [105.0, 68.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 68.0, 0.0]],
    [[105.0, 0.0, 0.0], [105.0, 68.0, 0.0]],
    [[0.0, 0.0, 0.0], [105.0, 0.0, 0.0]],
    [[0.0, 43.16, 0.0], [5.5, 43.16, 0.0]],
    [[5.5, 43.16, 0.0], [5.5, 24.84, 0.0]],
    [[5.5, 24.84, 0.0], [0.0, 24.84, 0.0]],
    [[99.5, 43.16, 0.0], [105.0, 43.16, 0.0]],
    [[99.5, 43.16, 0.0], [99.5, 24.84, 0.0]],
    [[99.5, 24.84, 0.0], [105.0, 24.84, 0.0]],
]


def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params["x_focal_length"]
    y_focal_length = cam_params["y_focal_length"]
    principal_point = np.array(cam_params["principal_point"])
    position_meters = np.array(cam_params["position_meters"])
    rotation = np.array(cam_params["rotation_matrix"])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array(
        [
            [x_focal_length, 0, principal_point[0]],
            [0, y_focal_length, principal_point[1]],
            [0, 0, 1],
        ]
    )
    P = Q @ (rotation @ It)

    return P


def inference(
    cam, frame, model, model_l, kp_threshold, line_threshold, device="cuda:0"
):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    frame = f.to_tensor(frame).float().unsqueeze(0)
    _, _, h_original, w_original = frame.size()
    frame = frame if frame.size()[-1] == 960 else transform2(frame)
    frame = frame.to(device)
    b, c, h, w = frame.size()

    with torch.no_grad():
        heatmaps = model(frame)
        heatmaps_l = model_l(frame)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
    kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
    lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
    final_dict = complete_keypoints(kp_dict, lines_dict, w=w, h=h, normalize=True)

    cam.update(final_dict[0])
    final_params_dict = cam.heuristic_voting()

    return final_params_dict


def project(frame, P):

    for line in lines_coords:
        w1 = line[0]
        w2 = line[1]
        i1 = P @ np.array([w1[0] - 105 / 2, w1[1] - 68 / 2, w1[2], 1])
        i2 = P @ np.array([w2[0] - 105 / 2, w2[1] - 68 / 2, w2[2], 1])
        i1 /= i1[-1]
        i2 /= i2[-1]
        frame = cv2.line(
            frame, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), (255, 0, 0), 3
        )

    r = 9.15
    pts1, pts2, pts3 = [], [], []
    base_pos = np.array([11 - 105 / 2, 68 / 2 - 68 / 2, 0.0, 0.0])
    for ang in np.linspace(37, 143, 50):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r * np.sin(ang), r * np.cos(ang), 0.0, 1.0])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts1.append([ipos[0], ipos[1]])

    base_pos = np.array([94 - 105 / 2, 68 / 2 - 68 / 2, 0.0, 0.0])
    for ang in np.linspace(217, 323, 200):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r * np.sin(ang), r * np.cos(ang), 0.0, 1.0])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts2.append([ipos[0], ipos[1]])

    base_pos = np.array([0, 0, 0.0, 0.0])
    for ang in np.linspace(0, 360, 500):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r * np.sin(ang), r * np.cos(ang), 0.0, 1.0])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts3.append([ipos[0], ipos[1]])

    XEllipse1 = np.array(pts1, np.int32)
    XEllipse2 = np.array(pts2, np.int32)
    XEllipse3 = np.array(pts2, np.int32)
    frame = cv2.polylines(frame, [XEllipse1], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [XEllipse2], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [XEllipse3], False, (255, 0, 0), 3)

    return frame


# Added by Victor Gaspar
def process_image_sequence(
    input_path,
    save_path,
    sequence_length,
    kp_threshold=0.1486,
    line_threshold=0.3880,
    weights_kp="SV_kp",
    weights_line="SV_lines",
    device="cuda:0",
    disable_tqdm=False,
):
    """Compute the camera parameters for a sequence of images and save them in a
    json file at save_path/cam_params.json.
    """

    this_file_path = os.path.dirname(os.path.realpath(__file__))

    cfg = yaml.safe_load(
        open(os.path.join(this_file_path, "config/hrnetv2_w48.yaml"), "r")
    )
    cfg_l = yaml.safe_load(
        open(os.path.join(this_file_path, "config/hrnetv2_w48_l.yaml"), "r")
    )

    loaded_state = torch.load(weights_kp, map_location=device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(device)
    model.eval()

    loaded_state_l = torch.load(weights_line, map_location=device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(device)
    model_l.eval()

    frame = cv2.imread(input_path + str(0).zfill(6) + ".png")
    cam = FramebyFrameCalib(
        iwidth=frame.shape[1], iheight=frame.shape[0], denormalize=True
    )
    all_cam_params = []
    for i in tqdm(
        range(sequence_length), disable=disable_tqdm, desc="Camera calibration"
    ):
        frame = cv2.imread(input_path + str(i).zfill(6) + ".png")
        final_params_dict = inference(
            cam, frame, model, model_l, kp_threshold, line_threshold, device=device
        )
        if final_params_dict is not None and final_params_dict["rep_err"] <= 25:
            all_cam_params.append(final_params_dict["cam_params"])
        else:
            all_cam_params.append(None)

    with open(os.path.join(save_path, "cam_params.json"), "w") as f:
        json.dump(all_cam_params, f)


def process_input(
    input_path,
    input_type,
    model_kp,
    model_line,
    kp_threshold,
    line_threshold,
    save_path,
    display,
):

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)

    if input_type == "video":
        cap = cv2.VideoCapture(input_path)
        if save_path != "":
            out = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
                fps,
                (frame_width, frame_height),
            )

        pbar = tqdm(total=total_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            final_params_dict = inference(
                cam, frame, model, model_l, kp_threshold, line_threshold
            )
            if final_params_dict is not None:
                P = projection_from_cam_params(final_params_dict)
                projected_frame = project(frame, P)
            else:
                projected_frame = frame

            if save_path != "":
                out.write(projected_frame)

            if display:
                cv2.imshow("Projected Frame", projected_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            pbar.update(1)

        cap.release()
        if save_path != "":
            out.release()
        cv2.destroyAllWindows()

    elif input_type == "image":
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Unable to read the image {input_path}")
            return

        final_params_dict = inference(
            cam, frame, model, model_l, kp_threshold, line_threshold
        )
        if final_params_dict is not None:
            P = projection_from_cam_params(final_params_dict)
            projected_frame = project(frame, P)
        else:
            projected_frame = frame

        if save_path != "":
            cv2.imwrite(save_path, projected_frame)
        else:
            plt.imshow(cv2.cvtColor(projected_frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process video or image and plot lines on each frame."
    )
    parser.add_argument(
        "--weights_kp", type=str, help="Path to the model for keypoint inference."
    )
    parser.add_argument(
        "--weights_line", type=str, help="Path to the model for line projection."
    )
    parser.add_argument(
        "--kp_threshold",
        type=float,
        default=0.1486,
        help="Threshold for keypoint detection.",
    )
    parser.add_argument(
        "--line_threshold",
        type=float,
        default=0.3880,
        help="Threshold for line detection.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="CPU or CUDA device index"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input video or image file.",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["video", "image"],
        required=True,
        help="Type of input: 'video' or 'image'.",
    )
    parser.add_argument(
        "--save_path", type=str, default="", help="Path to save the processed video."
    )
    parser.add_argument(
        "--display", action="store_true", help="Enable real-time display."
    )
    args = parser.parse_args()

    input_path = args.input_path
    input_type = args.input_type
    model_kp = args.weights_kp
    model_line = args.weights_line
    save_path = args.save_path
    device = args.device
    display = args.display and input_type == "video"
    kp_threshold = args.kp_threshold
    line_threshold = args.line_threshold

    cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml", "r"))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml", "r"))

    loaded_state = torch.load(args.weights_kp, map_location=device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(device)
    model.eval()

    loaded_state_l = torch.load(args.weights_line, map_location=device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(device)
    model_l.eval()

    transform2 = T.Resize((540, 960))

    process_input(
        input_path,
        input_type,
        model_kp,
        model_line,
        kp_threshold,
        line_threshold,
        save_path,
        display,
    )
