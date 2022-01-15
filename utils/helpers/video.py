import cv2
import sys
def get_all_true_box(list_bbox_path):
    boxes = []
    with open(list_bbox_path) as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines):
            elts = line.split()
            if len(elts) == 2:
                assert elts[1] == "0"
                boxes.append(None)
            else:
                assert len(elts) > 2
                assert elts[1] == "1"  # only one bbox per frame
                assert i == (int)(elts[0])  # frame number
                x = (int)(elts[2])
                y = (int)(elts[3])
                w = (int)(elts[4])
                h = (int)(elts[5])
                boxes.append((x, y, w, h))
    return boxes


def get_num_first_frame_with_coord(list_bbox_path):
    boxes = get_all_true_box(list_bbox_path)
    i = 0
    first = boxes[i]
    while first is None:
        i += 1
        first = boxes[i]
    return i, first


def get_frame_from_number(video_path, frame_nb):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: unable to read video:", video_path)
        sys.exit()

    frame_number = 1
    _, frame = cap.read()
    while frame_number != frame_nb:
        frame_number += 1
        _, frame = cap.read()
    return frame


def get_nb_frames_video(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length
