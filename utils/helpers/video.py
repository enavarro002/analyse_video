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

def compute_IOU(box1, box2) : 
    bb1_x1 = box1[0]
    bb1_y1 = box1[1]
    bb1_x2 = box1[0] + box1[2]
    bb1_y2 = box1[1] + box1[3]

    bb2_x1 = box2[0]
    bb2_y1 = box2[1]
    bb2_x2 = box2[0] + box2[2]
    bb2_y2 = box2[1] + box2[3]

    bb1 = [bb1_x1, bb1_y1, bb1_x2, bb1_y2]
    bb2 = [bb2_x1, bb2_y1, bb2_x2, bb2_y2]

    # bb1 va être représenté par 0 et bb2 par 1
    corresp = {0 : bb1, 1 : bb2}

    left_one = 0 if bb1_x1 <= bb2_x1 else 1
    right_one = 0 if bb1_x2 >= bb2_x2 else 1
    top_one = 0 if bb1_y1 <= bb2_y1 else 1
    bottom_one = 0 if bb1_y2 >= bb2_y2 else 1

    w_inter = corresp[1-right_one][2] - corresp[1-left_one][0]
    h_inter = corresp[1-bottom_one][3] - corresp[1-top_one][1]

    # cas où les boîtes ne se touchent pas 
    if(w_inter<0 or h_inter<0) : 
        return 0

    inter_area = w_inter*h_inter
    union_area = box1[2] * box1[3] + box2[2]* box2[3] - inter_area

    return (inter_area/union_area)