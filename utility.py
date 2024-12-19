import argparse
import os

import cv2
import numpy as np


def str2bool(v):
    return v.lower() in ("true", "yes", "t", "y", "1")


def init_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_angle_cls", type=str2bool, default=True)
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for text detector
    parser.add_argument("--det_model_path", type=str, default="models/ch_PP-OCRv4_det_infer.onnx")
    parser.add_argument("--det_limit_side_len", type=float, default=736)
    parser.add_argument("--det_limit_type", type=str, default="min")

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
    parser.add_argument("--det_db_max_candidates", type=int, default=1000)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    parser.add_argument("--use_dilation", type=str2bool, default=True)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # params for text recognizer
    parser.add_argument("--rec_model_path", type=str, default="models/ch_PP-OCRv4_rec_infer.onnx")
    parser.add_argument("--rec_image_shape", type=list, default=[3, 48, 320])
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--character_dict_path", type=str, default="./models/ppocr_keys_v1.txt")

    # params for text classifier

    parser.add_argument("--cls_model_path", type=str, default="models/ch_ppocr_mobile_v2.0_cls_infer.onnx")
    parser.add_argument("--cls_image_shape", type=list, default=[3, 48, 192])
    parser.add_argument("--label_list", type=list, default=["0", "180"])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    # extended function
    parser.add_argument(
        "--return_word_box",
        type=str2bool,
        default=False,
        help="Whether return the bbox of each word (split by space) or chinese character. Only used in ppstructure for layout recovery",
    )

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def create_predictor(args, model_file_path):
    import onnxruntime as ort

    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(model_file_path))
    if args.use_gpu:
        sess = ort.InferenceSession(
            model_file_path,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {"device_id": args.gpu_id, "cudnn_conv_algo_search": "DEFAULT"},
                )
            ],
        )
    else:
        sess = ort.InferenceSession(model_file_path, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0], None, None


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img
