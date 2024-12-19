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
import math
import time
from typing import List, Tuple, Union

import cv2
import numpy as np

import utility
from .postprocess import CTCLabelDecode


class TextRecognizer:
    def __init__(self, args):
        self.rec_batch_num = args.rec_batch_num
        self.rec_image_shape = args.rec_image_shape

        self.predictor, self.input_tensor, self.output_tensors, _ = utility.create_predictor(args, args.rec_model_path)

        self.postprocess_op = CTCLabelDecode(character_dict_path=args.character_dict_path, use_space_char=True)
        self.return_word_box = args.return_word_box

    def __call__(
        self, img_list: Union[np.ndarray, List[np.ndarray]], return_word_box: bool = False
    ) -> Tuple[List[Tuple[str, float]], float]:
        if isinstance(img_list, np.ndarray):
            img_list = [img_list]

        # Calculate the aspect ratio of all text bars
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        img_num = len(img_list)
        rec_res = [("", 0.0)] * img_num

        batch_num = self.rec_batch_num

        start_time = time.time()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)

            # Parameter Alignment for PaddleOCR
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)

            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img_batch.append(norm_img[np.newaxis, :])
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_dict = {self.input_tensor.name: norm_img_batch}
            outputs = self.predictor.run(self.output_tensors, input_dict)
            preds = outputs[0]

            rec_result = self.postprocess_op(
                preds,
                return_word_box=self.return_word_box,
                wh_ratio_list=wh_ratio_list,
                max_wh_ratio=max_wh_ratio,
            )

            for rno, one_res in enumerate(rec_result):
                rec_res[indices[beg_img_no + rno]] = one_res
        elapse = time.time() - start_time
        return rec_res, elapse

    def resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        img_channel, img_height, img_width = self.rec_image_shape
        assert img_channel == img.shape[2]

        img_width = int(img_height * max_wh_ratio)

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(img_height * ratio) > img_width:
            resized_w = img_width
        else:
            resized_w = int(math.ceil(img_height * ratio))

        resized_image = cv2.resize(img, (resized_w, img_height))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((img_channel, img_height, img_width), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
