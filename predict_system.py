import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import math

from .cls import TextClassifier
from .det import TextDetector
from .rec import TextRecognizer
from .utils import (
    LoadImage,
    UpdateParameters,
    VisRes,
    add_round_letterbox,
    get_logger,
    increase_min_side,
    init_args,
    read_yaml,
    reduce_max_side,
    update_model_path,
)

root_dir = Path(__file__).resolve().parent
DEFAULT_CFG_PATH = root_dir / "config.yaml"
logger = get_logger("RapidOCR")


class BoxedResult(object):
    """计算识别文字的汉字单字和英文单词的坐标框。代码借鉴自PaddlePaddle/PaddleOCR和fanqie03/char-detection"""

    def __init__(self):
        pass

    def __call__(
        self,
        imgs: Optional[List[np.ndarray]],
        dt_boxes: Optional[List[np.ndarray]],
        rec_res: Optional[List[Any]],
    ):
        res = []
        for img, box, rec_res in zip(imgs, dt_boxes, rec_res):
            direction = self.get_box_direction(box)

            rec_txt, rec_conf, rec_word_info = rec_res[0], rec_res[1], rec_res[2]
            h, w = img.shape[:2]
            img_box = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            word_box_content_list, word_box_list, conf_list = self.cal_ocr_word_box(
                rec_txt, img_box, rec_word_info
            )
            word_box_list = self.adjust_box_overlap(copy.deepcopy(word_box_list))
            word_box_list = self.reverse_rotate_crop_image(
                copy.deepcopy(box), word_box_list, direction
            )
            res.append(
                [rec_txt, rec_conf, word_box_list, word_box_content_list, conf_list]
            )
        return res

    @staticmethod
    def get_box_direction(box: np.ndarray) -> str:
        direction = "w"
        img_crop_width = int(
            max(
                np.linalg.norm(box[0] - box[1]),
                np.linalg.norm(box[2] - box[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(box[0] - box[3]),
                np.linalg.norm(box[1] - box[2]),
            )
        )
        if img_crop_height * 1.0 / img_crop_width >= 1.5:
            direction = "h"
        return direction

    @staticmethod
    def cal_ocr_word_box(
        rec_txt: str, box: np.ndarray, rec_word_info: List[Tuple[str, List[int]]]
    ) -> Tuple[List[str], List[List[int]], List[float]]:
        """Calculate the detection frame for each word based on the results of recognition and detection of ocr
        汉字坐标是单字的
        英语坐标是单词级别的
        """

        col_num, word_list, word_col_list, state_list, conf_list = rec_word_info
        box = box.tolist()
        bbox_x_start = box[0][0]
        bbox_x_end = box[1][0]
        bbox_y_start = box[0][1]
        bbox_y_end = box[2][1]

        cell_width = (bbox_x_end - bbox_x_start) / col_num
        word_box_list = []
        word_box_content_list = []
        cn_width_list = []
        en_width_list = []
        cn_col_list = []
        en_col_list = []

        def cal_char_width(width_list, word_col_):
            if len(word_col_) == 1:
                return
            char_total_length = (word_col_[-1] - word_col_[0]) * cell_width
            char_width = char_total_length / (len(word_col_) - 1)
            width_list.append(char_width)

        def cal_box(col_list, width_list, word_box_list_):
            if len(col_list) == 0:
                return
            if len(width_list) != 0:
                avg_char_width = np.mean(width_list)
            else:
                avg_char_width = (bbox_x_end - bbox_x_start) / len(rec_txt)

            for center_idx in col_list:
                center_x = (center_idx + 0.5) * cell_width
                cell_x_start = max(int(center_x - avg_char_width / 2), 0) + bbox_x_start
                cell_x_end = (
                    min(int(center_x + avg_char_width / 2), bbox_x_end - bbox_x_start)
                    + bbox_x_start
                )
                cell = [
                    [cell_x_start, bbox_y_start],
                    [cell_x_end, bbox_y_start],
                    [cell_x_end, bbox_y_end],
                    [cell_x_start, bbox_y_end],
                ]
                word_box_list_.append(cell)

        for word, word_col, state in zip(word_list, word_col_list, state_list):
            if state == "cn":
                cal_char_width(cn_width_list, word_col)
                cn_col_list += word_col
                word_box_content_list += word
            else:
                cal_char_width(en_width_list, word_col)
                en_col_list += word_col
                word_box_content_list += word

        cal_box(cn_col_list, cn_width_list, word_box_list)
        cal_box(en_col_list, en_width_list, word_box_list)
        sorted_word_box_list = sorted(word_box_list, key=lambda box: box[0][0])
        return word_box_content_list, sorted_word_box_list, conf_list

    @staticmethod
    def adjust_box_overlap(
        word_box_list: List[List[List[int]]],
    ) -> List[List[List[int]]]:
        # 调整bbox有重叠的地方
        for i in range(len(word_box_list) - 1):
            cur, nxt = word_box_list[i], word_box_list[i + 1]
            if cur[1][0] > nxt[0][0]:  # 有交集
                distance = abs(cur[1][0] - nxt[0][0])
                cur[1][0] -= distance / 2
                cur[2][0] -= distance / 2
                nxt[0][0] += distance - distance / 2
                nxt[3][0] += distance - distance / 2
        return word_box_list

    def reverse_rotate_crop_image(
        self,
        bbox_points: np.ndarray,
        word_points_list: List[List[List[int]]],
        direction: str = "w",
    ) -> List[List[List[int]]]:
        """
        get_rotate_crop_image的逆操作
        img为原图
        part_img为crop后的图
        bbox_points为part_img中对应在原图的bbox, 四个点，左上，右上，右下，左下
        part_points为在part_img中的点[(x, y), (x, y)]
        """
        bbox_points = np.float32(bbox_points)

        left = int(np.min(bbox_points[:, 0]))
        top = int(np.min(bbox_points[:, 1]))
        bbox_points[:, 0] = bbox_points[:, 0] - left
        bbox_points[:, 1] = bbox_points[:, 1] - top

        img_crop_width = int(np.linalg.norm(bbox_points[0] - bbox_points[1]))
        img_crop_height = int(np.linalg.norm(bbox_points[0] - bbox_points[3]))

        pts_std = np.array(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        ).astype(np.float32)
        M = cv2.getPerspectiveTransform(bbox_points, pts_std)
        _, IM = cv2.invert(M)

        new_word_points_list = []
        for word_points in word_points_list:
            new_word_points = []
            for point in word_points:
                new_point = point
                if direction == "h":
                    new_point = self.s_rotate(
                        math.radians(-90), new_point[0], new_point[1], 0, 0
                    )
                    new_point[0] = new_point[0] + img_crop_width

                p = np.float32(new_point + [1])
                x, y, z = np.dot(IM, p)
                new_point = [x / z, y / z]

                new_point = [int(new_point[0] + left), int(new_point[1] + top)]
                new_word_points.append(new_point)
            new_word_points = self.order_points(new_word_points)
            new_word_points_list.append(new_word_points)
        return new_word_points_list

    @staticmethod
    def s_rotate(angle, valuex, valuey, pointx, pointy):
        """绕pointx,pointy顺时针旋转
        https://blog.csdn.net/qq_38826019/article/details/84233397
        """
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        sRotatex = (
            (valuex - pointx) * math.cos(angle)
            + (valuey - pointy) * math.sin(angle)
            + pointx
        )
        sRotatey = (
            (valuey - pointy) * math.cos(angle)
            - (valuex - pointx) * math.sin(angle)
            + pointy
        )
        return [sRotatex, sRotatey]

    @staticmethod
    def order_points(box: List[List[int]]) -> List[List[int]]:
        """矩形框顺序排列"""

        def convert_to_1x2(p):
            if p.shape == (2,):
                return p.reshape((1, 2))
            elif p.shape == (1, 2):
                return p
            else:
                return p[:1, :]

        box = np.array(box).reshape((-1, 2))
        center_x, center_y = np.mean(box[:, 0]), np.mean(box[:, 1])
        if np.any(box[:, 0] == center_x) and np.any(
            box[:, 1] == center_y
        ):  # 有两点横坐标相等，有两点纵坐标相等，菱形
            p1 = box[np.where(box[:, 0] == np.min(box[:, 0]))]
            p2 = box[np.where(box[:, 1] == np.min(box[:, 1]))]
            p3 = box[np.where(box[:, 0] == np.max(box[:, 0]))]
            p4 = box[np.where(box[:, 1] == np.max(box[:, 1]))]
        elif np.all(box[:, 0] == center_x):  # 四个点的横坐标都相同
            y_sort = np.argsort(box[:, 1])
            p1 = box[y_sort[0]]
            p2 = box[y_sort[1]]
            p3 = box[y_sort[2]]
            p4 = box[y_sort[3]]
        elif np.any(box[:, 0] == center_x) and np.all(
            box[:, 1] != center_y
        ):  # 只有两点横坐标相等，先上下再左右
            p12, p34 = (
                box[np.where(box[:, 1] < center_y)],
                box[np.where(box[:, 1] > center_y)],
            )
            p1, p2 = (
                p12[np.where(p12[:, 0] == np.min(p12[:, 0]))],
                p12[np.where(p12[:, 0] == np.max(p12[:, 0]))],
            )
            p3, p4 = (
                p34[np.where(p34[:, 0] == np.max(p34[:, 0]))],
                p34[np.where(p34[:, 0] == np.min(p34[:, 0]))],
            )
        else:  # 只有两点纵坐标相等，或者是没有相等的，先左右再上下
            p14, p23 = (
                box[np.where(box[:, 0] < center_x)],
                box[np.where(box[:, 0] > center_x)],
            )
            p1, p4 = (
                p14[np.where(p14[:, 1] == np.min(p14[:, 1]))],
                p14[np.where(p14[:, 1] == np.max(p14[:, 1]))],
            )
            p2, p3 = (
                p23[np.where(p23[:, 1] == np.min(p23[:, 1]))],
                p23[np.where(p23[:, 1] == np.max(p23[:, 1]))],
            )

        # 解决单字切割后横坐标完全相同的shape错误
        p1 = convert_to_1x2(p1)
        p2 = convert_to_1x2(p2)
        p3 = convert_to_1x2(p3)
        p4 = convert_to_1x2(p4)
        return np.array([p1, p2, p3, p4]).reshape((-1, 2)).tolist()


class TextSystem:
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        if config_path is not None and Path(config_path).exists():
            config = read_yaml(config_path)
        else:
            config = read_yaml(DEFAULT_CFG_PATH)
        config = update_model_path(config)

        if kwargs:
            updater = UpdateParameters()
            config = updater(config, **kwargs)

        global_config = config["Global"]
        self.print_verbose = global_config["print_verbose"]
        self.text_score = global_config["text_score"]
        self.min_height = global_config["min_height"]
        self.width_height_ratio = global_config["width_height_ratio"]

        self.use_det = global_config["use_det"]
        self.text_det = TextDetector(config["Det"])

        self.use_cls = global_config["use_cls"]
        self.text_cls = TextClassifier(config["Cls"])

        self.use_rec = global_config["use_rec"]
        self.text_rec = TextRecognizer(config["Rec"])

        self.load_img = LoadImage()
        self.max_side_len = global_config["max_side_len"]
        self.min_side_len = global_config["min_side_len"]

        self.cal_rec_boxes = BoxedResult()

    def __call__(
        self,
        img_content: Union[str, np.ndarray, bytes, Path],
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[Optional[List[List[Union[Any, str]]]], Optional[List[float]]]:
        use_det = self.use_det if use_det is None else use_det
        use_cls = self.use_cls if use_cls is None else use_cls
        use_rec = self.use_rec if use_rec is None else use_rec
        return_word_box = False
        if kwargs:
            box_thresh = kwargs.get("box_thresh", 0.5)
            unclip_ratio = kwargs.get("unclip_ratio", 1.6)
            text_score = kwargs.get("text_score", 0.5)
            return_word_box = kwargs.get("return_word_box", False)
            self.text_det.postprocess_op.box_thresh = box_thresh
            self.text_det.postprocess_op.unclip_ratio = unclip_ratio
            self.text_score = text_score

        img = self.load_img(img_content)

        raw_h, raw_w = img.shape[:2]
        op_record = {}
        img, ratio_h, ratio_w = self.preprocess(img)
        op_record["preprocess"] = {"ratio_h": ratio_h, "ratio_w": ratio_w}

        dt_boxes, cls_res, rec_res = None, None, None
        det_elapse, cls_elapse, rec_elapse = 0.0, 0.0, 0.0

        if use_det:
            img, op_record = self.maybe_add_letterbox(img, op_record)
            dt_boxes, det_elapse = self.auto_text_det(img)
            if dt_boxes is None:
                return None, None

            img = self.get_crop_img_list(img, dt_boxes)

        if use_cls:
            img, cls_res, cls_elapse = self.text_cls(img)

        if use_rec:
            rec_res, rec_elapse = self.text_rec(img, return_word_box)

        if dt_boxes is not None and rec_res is not None and return_word_box:
            rec_res = self.cal_rec_boxes(img, dt_boxes, rec_res)
            for rec_res_i in rec_res:
                if rec_res_i[2]:
                    rec_res_i[2] = (
                        self._get_origin_points(rec_res_i[2], op_record, raw_h, raw_w)
                        .astype(np.int32)
                        .tolist()
                    )

        if dt_boxes is not None:
            dt_boxes = self._get_origin_points(dt_boxes, op_record, raw_h, raw_w)

        ocr_res = self.get_final_res(
            dt_boxes, cls_res, rec_res, det_elapse, cls_elapse, rec_elapse
        )
        return ocr_res

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        h, w = img.shape[:2]
        max_value = max(h, w)
        ratio_h = ratio_w = 1.0
        if max_value > self.max_side_len:
            img, ratio_h, ratio_w = reduce_max_side(img, self.max_side_len)

        h, w = img.shape[:2]
        min_value = min(h, w)
        if min_value < self.min_side_len:
            img, ratio_h, ratio_w = increase_min_side(img, self.min_side_len)
        return img, ratio_h, ratio_w

    def maybe_add_letterbox(
        self, img: np.ndarray, op_record: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = img.shape[:2]

        if self.width_height_ratio == -1:
            use_limit_ratio = False
        else:
            use_limit_ratio = w / h > self.width_height_ratio

        if h <= self.min_height or use_limit_ratio:
            padding_h = self._get_padding_h(h, w)
            block_img = add_round_letterbox(img, (padding_h, padding_h, 0, 0))
            op_record["padding_1"] = {"top": padding_h, "left": 0}
            return block_img, op_record

        op_record["padding_1"] = {"top": 0, "left": 0}
        return img, op_record

    def _get_padding_h(self, h: int, w: int) -> int:
        new_h = max(int(w / self.width_height_ratio), self.min_height) * 2
        padding_h = int(abs(new_h - h) / 2)
        return padding_h

    def auto_text_det(
        self, img: np.ndarray
    ) -> Tuple[Optional[List[np.ndarray]], float]:
        dt_boxes, det_elapse = self.text_det(img)
        if dt_boxes is None or len(dt_boxes) < 1:
            return None, 0.0

        dt_boxes = self.sorted_boxes(dt_boxes)
        return dt_boxes, det_elapse

    def get_crop_img_list(
        self, img: np.ndarray, dt_boxes: List[np.ndarray]
    ) -> List[np.ndarray]:
        def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
            img_crop_width = int(
                max(
                    np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[2] - points[3]),
                )
            )
            img_crop_height = int(
                max(
                    np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2]),
                )
            )
            pts_std = np.array(
                [
                    [0, 0],
                    [img_crop_width, 0],
                    [img_crop_width, img_crop_height],
                    [0, img_crop_height],
                ]
            ).astype(np.float32)
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

        img_crop_list = []
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        return img_crop_list

    @staticmethod
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
                if (
                    abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10
                    and _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def _get_origin_points(
        self,
        dt_boxes: List[np.ndarray],
        op_record: Dict[str, Any],
        raw_h: int,
        raw_w: int,
    ) -> np.ndarray:
        dt_boxes_array = np.array(dt_boxes).astype(np.float32)
        for op in reversed(list(op_record.keys())):
            v = op_record[op]
            if "padding" in op:
                top, left = v.get("top"), v.get("left")
                dt_boxes_array[:, :, 0] -= left
                dt_boxes_array[:, :, 1] -= top
            elif "preprocess" in op:
                ratio_h = v.get("ratio_h")
                ratio_w = v.get("ratio_w")
                dt_boxes_array[:, :, 0] *= ratio_w
                dt_boxes_array[:, :, 1] *= ratio_h

        dt_boxes_array = np.where(dt_boxes_array < 0, 0, dt_boxes_array)
        dt_boxes_array[..., 0] = np.where(
            dt_boxes_array[..., 0] > raw_w, raw_w, dt_boxes_array[..., 0]
        )
        dt_boxes_array[..., 1] = np.where(
            dt_boxes_array[..., 1] > raw_h, raw_h, dt_boxes_array[..., 1]
        )
        return dt_boxes_array

    def get_final_res(
        self,
        dt_boxes: Optional[List[np.ndarray]],
        cls_res: Optional[List[List[Union[str, float]]]],
        rec_res: Optional[List[Tuple[str, float, List[Union[str, float]]]]],
        det_elapse: float,
        cls_elapse: float,
        rec_elapse: float,
    ) -> Tuple[Optional[List[List[Union[Any, str]]]], Optional[List[float]]]:
        if dt_boxes is None and rec_res is None and cls_res is not None:
            return cls_res, [cls_elapse]

        if dt_boxes is None and rec_res is None:
            return None, None

        if dt_boxes is None and rec_res is not None:
            return [[res[0], res[1]] for res in rec_res], [rec_elapse]

        if dt_boxes is not None and rec_res is None:
            return [box.tolist() for box in dt_boxes], [det_elapse]

        dt_boxes, rec_res = self.filter_result(dt_boxes, rec_res)
        if not dt_boxes or not rec_res or len(dt_boxes) <= 0:
            return None, None

        ocr_res = [[box.tolist(), *res] for box, res in zip(dt_boxes, rec_res)], [
            det_elapse,
            cls_elapse,
            rec_elapse,
        ]
        return ocr_res

    def filter_result(
        self,
        dt_boxes: Optional[List[np.ndarray]],
        rec_res: Optional[List[Tuple[str, float]]],
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[Tuple[str, float]]]]:
        if dt_boxes is None or rec_res is None:
            return None, None

        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt[0], rec_reuslt[1]
            if float(score) >= self.text_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)

        return filter_boxes, filter_rec_res


def main():
    args = init_args()
    ocr_engine = TextSystem(**vars(args))

    use_det = not args.no_det
    use_cls = not args.no_cls
    use_rec = not args.no_rec
    result, elapse_list = ocr_engine(
        args.img_path,
        use_det=use_det,
        use_cls=use_cls,
        use_rec=use_rec,
        **vars(args)
    )
    logger.info(result)

    if args.print_cost:
        logger.info(elapse_list)

    if args.vis_res:
        vis = VisRes()
        Path(args.vis_save_path).mkdir(parents=True, exist_ok=True)
        save_path = Path(args.vis_save_path) / f"{Path(args.img_path).stem}_vis.png"

        if use_det and not use_cls and not use_rec:
            boxes, *_ = list(zip(*result))
            vis_img = vis(args.img_path, boxes)
            cv2.imwrite(str(save_path), vis_img)
            logger.info("The vis result has saved in %s", save_path)

        elif use_det and use_rec:
            font_path = Path(args.vis_font_path)
            if not font_path.exists():
                raise FileExistsError(f"{font_path} does not exist!")

            boxes, txts, scores = list(zip(*result))
            vis_img = vis(args.img_path, boxes, txts, scores, font_path=font_path)
            cv2.imwrite(str(save_path), vis_img)
            logger.info("The vis result has saved in %s", save_path)


if __name__ == "__main__":
    main()
