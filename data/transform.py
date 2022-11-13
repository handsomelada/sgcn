import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

def translate(poses, bbox):
    if len(bbox) != 0:
        bbox_x, bbox_y = bbox[0], bbox[1]
        poses[..., 0] = (poses[..., 0] - bbox_x)
        poses[..., 1] = (poses[..., 1] - bbox_y)
    else:
        pass
    return poses


def Normalize(poses, bbox):
    h_bbox, w_bbox = bbox[3], bbox[2]
    poses[..., 0] = (poses[..., 0] / w_bbox)
    poses[..., 1] = (poses[..., 1] / h_bbox)
    return poses

def rand_scale(poses, bbox, thr,img_shape=[720, 1280]):
    img_h, img_w = img_shape[0], img_shape[1]
    rand_factor = random.random()
    if len(bbox) != 0:
        x1,y1,w,h = bbox
        fa = w / img_w
        if fa > 0.15:
            if rand_factor >= thr:
                poses[..., 0] = (poses[..., 0] * rand_factor)
                poses[..., 1] = (poses[..., 1] * rand_factor)
                return poses
            else:
                return poses
        else:
            if rand_factor >= 0.7:
                poses[..., 0] = (poses[..., 0] / rand_factor)
                poses[..., 1] = (poses[..., 1] / rand_factor)
                return poses
            else:
                return poses
    else:
        return poses

def poseInimg_flip(poses, mode, img_shape=[720, 1280]):
    h, w = img_shape[0], img_shape[1]
    if mode == 'horizontal':
        poses[..., 0] = (w - poses[..., 0])
    elif mode == 'vertical':
        poses[..., 1] = (h - poses[..., 1])
    else:
        pass
    return poses

def poseInbbox_flip(poses, bbox, mode):
    if len(bbox) != 0:
        x1,y1,w,h = bbox
        xm = x1 + w/2
        ym = y1 + h/2
        if mode == 'horizontal':
            poses[..., 0] = (2*xm - poses[..., 0])
        elif mode == 'vertical':
            poses[..., 1] = (2*ym - poses[..., 1])
        else:
            pass
    else:
        pass
    return poses

def PreNormalize2D(results):
    img_shape = [720, 1280]
    h, w = img_shape[0], img_shape[1]
    results[..., 0] = (results[..., 0] - (w / 2)) / (w / 2)
    results[..., 1] = (results[..., 1] - (h / 2)) / (h / 2)
    return results



if __name__ == '__main__':

    bbox= [489.3,460.,404.1, 127.1]
    kpts = [
     [793, 490, 1.],
     [804, 528, 1.],
     [831, 482, 1.],
     [845, 584, 1.],
     [874, 510, 1.],
     [874, 559, 1.],
     [671, 500, 1.],
     [673, 528, 1.],
    [588, 507, 1.],
    [585, 536, 1.],
    [514, 477, 1.],
    [500, 529, 1.]
    ]

    bbox_lying=[777,509.25,897,720]
    w,h = 120, 210.75
    kpts_lying = [
    [855.5, 720, 0.054901],
        [798.5, 713, 0.096863],
        [866.5, 697, 0.074219],
        [776, 688, 0.15259],
        [870, 694, 0.05899],
        [782, 703, 0.10596],
        [864.5, 644.5, 0.82031],
        [826.5, 639.5, 0.87012],
        [866.5, 598.5, 0.92725],
        [828.5, 591.5, 0.94971],
        [877, 542.5, 0.96729],
        [839.5, 538, 0.97461]]


    kpts_lying = np.array(kpts_lying)

    fig = plt.figure(figsize=(16, 8))
    plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False


    plt.title("vis keypoints")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(kpts_lying[:, 0], kpts_lying[:, 1], s=20, c='b', marker='*')
    plt.scatter(bbox_lying[0], bbox_lying[1], s=20, c='b', marker='o')
    plt.scatter(bbox_lying[2], bbox_lying[3], s=20, c='b', marker='o')

    # for a, b in zip(kpts[:,0], kpts[:,1]):
    #     plt.text(a, b, (a, b))




    # kpts_after_flip = poseInbbox_flip(kpts, bbox, 'horizontal')
    # kpts_after_flip = poseInimg_flip(kpts, 'horizontal', img_shape=[720, 1080])
    kpts_after_scale = rand_scale(kpts_lying, bbox_lying, 0.5)
    plt.scatter(kpts_after_scale[:, 0], kpts_after_scale[:, 1], s=50, c='r', marker='o')

    # for a, b in zip(kpts_after_flip[:,0], kpts_after_flip[:,1]):
    #     plt.text(a, b, (a, b))

    fig.savefig('vis_points.jpg', dpi=100)
    fig.clf()
    plt.close()
    print('finished!')






