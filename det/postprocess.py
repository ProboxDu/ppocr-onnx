from typing import List, Tuple, Dict

import cv2
import numpy as np
import pyclipper
from numpy import ndarray
from shapely.geometry import Polygon


class DBPostProcess:
    """The post process for Differentiable Binarization (DB)."""

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.7,
        max_candidates: int = 1000,
        unclip_ratio: float = 2.0,
        score_mode: str = "fast",
        use_dilation: bool = False,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode

        self.dilation_kernel = None

        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])

    def __call__(self, outs_dict: Dict[str, ndarray], shape_list: List[Tuple[int, int, float, float]]) -> List[Dict[str, ndarray]]:
        pred = outs_dict["maps"]
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(np.array(segmentation[batch_index]).astype(np.uint8), self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)

            boxes_batch.append({"points": boxes})
        return boxes_batch

    def boxes_from_bitmap(self, pred: np.ndarray, bitmap: np.ndarray, dest_width: int, dest_height: int) -> Tuple[np.ndarray, List[float]]:
        """
        bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        """

        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes, scores = [], []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)

            if self.box_thresh > score:
                continue

            box = self.unclip(points)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int32))
            scores.append(score)
        return np.array(boxes, dtype=np.int32), scores

    @staticmethod
    def get_mini_boxes(contour: np.ndarray) -> Tuple[np.ndarray, float]:
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0

        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = np.array([points[index_1], points[index_2], points[index_3], points[index_4]])
        return box, min(bounding_box[1])

    @staticmethod
    def box_score_fast(bitmap: np.ndarray, _box: np.ndarray) -> float:
        """
        box_score_fast: use bbox mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    @staticmethod
    def box_score_slow(bitmap: np.ndarray, contour: np.ndarray) -> float:
        """
        box_score_slow: use polyon mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def unclip(self, box: np.ndarray) -> np.ndarray:
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance)).reshape((-1, 1, 2))
        return expanded
