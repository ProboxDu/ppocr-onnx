import copy
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from cls import TextClassifier
from det import TextDetector
from rec import TextRecognizer
from utility import get_rotate_crop_image
from utils import get_logger

logger = get_logger("ppocr-onnx")


class TextSystem(object):
    def __init__(self, args):
        self.args = args
        self.text_detector = TextDetector(args)
        self.text_recognizer = TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score

        if self.use_angle_cls:
            self.text_classifier = TextClassifier(args)

    def __call__(self, img: np.ndarray) -> Tuple[Optional[List[List[Union[Any, str]]]], Optional[List[float]]]:
        det_elapse, cls_elapse, rec_elapse = 0.0, 0.0, 0.0
        dt_boxes, det_elapse = self.text_detector(img)
        if dt_boxes is None or len(dt_boxes) < 1:
            return None, [det_elapse, cls_elapse, rec_elapse]

        dt_boxes = sorted_boxes(dt_boxes)

        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)

        if self.use_angle_cls:
            img_crop_list, cls_res, cls_elapse = self.text_classifier(img_crop_list)

        rec_res, rec_elapse = self.text_recognizer(img_crop_list)

        dt_boxes, rec_res = self.filter_results(dt_boxes, rec_res)
        ocr_res = [[box.tolist(), *rec] for box, rec in zip(dt_boxes, rec_res)], [det_elapse, cls_elapse, rec_elapse]
        return ocr_res

    def filter_results(
        self, dt_boxes: Optional[List[np.ndarray]], rec_res: Optional[List[Tuple[str, float]]]
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[Tuple[str, float]]]]:
        if dt_boxes is None or rec_res is None:
            return None, None

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if float(score) >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes: np.ndarray) -> List[np.ndarray]:
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and _boxes[j + 1][0][0] < _boxes[j][0][0]:
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes
