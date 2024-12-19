import logging
import sys

import cv2

import utility
from predict_system import TextSystem


def main():
    ocr = TextSystem(args)
    img = cv2.imread("test.png")

    img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    res = ocr(img)

    for boxed_result in res:
        print(boxed_result)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    args = utility.parse_args()
    main()
