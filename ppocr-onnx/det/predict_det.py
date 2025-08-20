# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from typing import Optional, Tuple

import numpy as np

import utility
from .postprocess import DBPostProcess
from .preprocess import create_operators, transform


class TextDetector:
    def __init__(self, args):
        pre_process_list = [
            {
                "DetResizeForTest": {
                    "limit_side_len": args.det_limit_side_len,
                    "limit_type": args.det_limit_type,
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        self.preprocess_op = create_operators(pre_process_list)

        post_process = {
            "thresh": args.det_db_thresh,
            "box_thresh": args.det_db_box_thresh,
            "max_candidates": args.det_db_max_candidates,
            "unclip_ratio": args.det_db_unclip_ratio,
            "use_dilation": args.use_dilation,
            "score_mode": args.det_db_score_mode,
        }
        self.postprocess_op = DBPostProcess(**post_process)

        self.predictor, self.input_tensor, self.output_tensors, _ = utility.create_predictor(args, args.det_model_path)

    def __call__(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        ori_img_shape = img.shape[0], img.shape[1]
        data = {"image": img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0

        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        start_time = time.time()

        input_dict = {self.input_tensor.name: img}

        outputs = self.predictor.run(self.output_tensors, input_dict)

        preds = {"maps": outputs[0]}
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_img_shape)
        elapse = time.time() - start_time
        return dt_boxes, elapse

    def filter_tag_det_res(self, dt_boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        img_height, img_width = image_shape
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)

            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue

            dt_boxes_new.append(box)
        return np.array(dt_boxes_new)

    @staticmethod
    def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-coordinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    @staticmethod
    def clip_det_res(points: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points
