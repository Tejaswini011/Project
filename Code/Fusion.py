import glob
import cv2
import numpy as np
import Fusion as fu
import struct
import matplotlib.pyplot as plt
import open3d as o3d
import random
from scipy.optimize import linear_sum_assignment
import statistics as st

class LiDAR2Camera(object):
    def __init__(self, calib_file):
        calibs = self.read_calib_file(calib_file)
        P = calibs["P2"]
        self.P = np.reshape(P, (3, 4))
        # Rigid transform from Lidar coord to reference camera coord
        V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(V2C, (3, 4))
        # Rotation from reference camera coord to rect camera coord
        R0 = calibs["R0_rect"]
        self.R0 = np.reshape(R0, (3, 3))

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

class FusedObject(object):
    def __init__(self, bbox2d, bbox3d, category, t, confidence):
        self.bbox2d = bbox2d
        self.bbox3d = bbox3d
        self.category = category
        self.confidence = confidence
        self.t = t
        with open("..//Data//model//yolov4/coco.names", 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        self.class_ = classes[category]

class Object2D(object):
    def __init__(self, box):
        self.category = int(box[0])
        self.confidence = box[2]
        self.xmin = int(box[1][0])
        self.ymin = int(box[1][1])
        self.xmax = int(box[1][0] + box[1][2])
        self.ymax = int(box[1][1] + box[1][3])
        self.bbox = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

class Object3D(object):
    """ 3d object label """
    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.bbox2d = np.zeros(shape=(2,2))
        self.bbox3d = np.zeros(shape=(4,2))

def load_data(root_dir):
    image_files = sorted(glob.glob(root_dir+"/images/*.png"))
    point_files = sorted(glob.glob(root_dir+"/points/*.pcd"))
    label_files = sorted(glob.glob(root_dir+"/labels/*.txt"))
    calib_files = sorted(glob.glob(root_dir+"/calibs/*.txt"))

    return image_files, point_files, label_files, calib_files

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [fu.Object3D(line) for line in lines if line.split(" ")[0]!="DontCare"]
    return objects


def fill_2D_obstacles(detections):
    return [fu.Object2D(box) for box in detections]


def get_rotation_mat(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    # rotation matrix about Y-axis
    # since in camera frame vertical is Y-axis and yaw angle in about the vertical axis
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def convert_3d_to_homo(pts_3d):
    """ Input: nx3 points in Cartesian
        Output: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def convert_3DBox_to_2DBox(pts_2d):
    x0 = np.min(pts_2d[:, 0])
    x1 = np.max(pts_2d[:, 0])
    y0 = np.min(pts_2d[:, 1])
    y1 = np.max(pts_2d[:, 1])
    x0 = max(0, x0)
    y0 = max(0, y0)
    return np.array([x0, y0, x1, y1])


def get_iou(box1, box2):
    """
    Computer Intersection Over Union
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = (box1_area + box2_area) - inter_area

    # compute the IoU
    iou = inter_area/float(union_area)
    return iou


def draw_projected_box2d(image, qs, color=(255,0,0), thickness=2):
    return cv2.rectangle(image, (int(qs[0]), int(qs[1])), (int(qs[2]), int(qs[3])), color, thickness)


def draw_projected_box3d(image, qs, color=(255, 0, 0), thickness=2):
    """
    Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
         1 --------- 0
        /|          /|
        2 -------- 3 |
        | |        | |
        | 5 -------| 4
        |/         |/
        6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

    return image

#Lidar Utilities

def project_to_image(l2c_object, pts_3d):
    """ Project 3d points to image plane.
    """
    # Convert to Homogeneous Coordinates
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # Multiply with the P Matrix
    pts_2d = np.dot(pts_3d_extend, np.transpose(l2c_object.P))  # nx3
    # Convert Back to Cartesian
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    return pts_2d[:, 0:2]


def compute_box_3d(l2c_object, obj):
    """ Projects the 3d bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = get_rotation_mat(obj.ry)
    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    # multiply the rotation matrix with the points
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

    # perform translation in all axis
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]

    # only draw 3d bounding box for objs in front of the camera
    # corners_3d[2, :] = z (distance) value from the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(l2c_object, np.transpose(corners_3d))

    return corners_2d


def get_image_with_bboxes(l2c_object, img, objects):
    img2 = img.copy()
    img3 = img.copy()
    for obj in objects:
        boxes = compute_box_3d(l2c_object, obj)
        if boxes is not None:
            obj.bbox3d = boxes
            obj.bbox2d = convert_3DBox_to_2DBox(boxes)
            img2 = draw_projected_box2d(img2, obj.bbox2d) # Draw the 2D Bounding Box
            img3 = draw_projected_box3d(img3, obj.bbox3d) # Draw the 3D Bounding Box
    return img2, img3


def convert_3D_to_2D(l2c_object, pts_3d_velo, print_values=False):
    '''
    Input: 3D points in Velodyne Frame [nx3]
    Output: 2D Pixels in Image Frame [nx2]
    '''
    R0_homo = np.vstack((l2c_object.R0, [0,0,0]))
    if print_values:
        print("R0_homo ", R0_homo)

    R0_homo_2 = np.hstack((R0_homo, [[0],[0],[0],[1]]))
    if print_values:
        print("R0_homo_2 ", R0_homo_2)

    P_R0 = np.dot(l2c_object.P, R0_homo_2)
    if print_values:
        print("P_R0 ", P_R0)

    P_R0_Rt = np.dot(P_R0, np.vstack((l2c_object.V2C, [0, 0, 0, 1])))
    if print_values:
        print("P_R0_Rt ", P_R0_Rt)

    pts_3d_homo = np.hstack((pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))))
    if print_values:
        print("pts_3d_homo ", pts_3d_homo)

    P_R0_Rt_X = np.dot(P_R0_Rt, pts_3d_homo.T)
    if print_values:
        print("P_R0_Rt_X ", P_R0_Rt_X)

    pts_2d_homo = P_R0_Rt_X.T
    if print_values:
        print("pts_2d_homo ", pts_2d_homo)

    pts_2d_homo[:, 0] /= pts_2d_homo[:, 2]
    pts_2d_homo[:, 1] /= pts_2d_homo[:, 2]

    pts_2d = pts_2d_homo[:, :2]

    if print_values:
        print("pts_2d ", pts_2d)

    return pts_2d


def remove_lidar_points_beyond_img(l2c_object, pts_3d, xmin, ymin, xmax, ymax, clip_distance=2.0):
    """ Filter lidar points, keep only those which lie inside image """
    pts_2d = convert_3D_to_2D(l2c_object, pts_3d)
    inside_pts_indices = ((pts_2d[:, 0] >= xmin) & (pts_2d[:, 0] < xmax) & (pts_2d[:, 1] >= ymin) & (pts_2d[:, 1] < ymax))

    # pc_velo are the points in LiDAR frame
    # therefore x-axis is in forward direction
    # we want to keep objects that are at least clip_distance away from sensor
    # X points are at 0th index column
    inside_pts_indices = inside_pts_indices & (pts_3d[:, 0] > clip_distance)
    pts_3d_inside_img = pts_3d[inside_pts_indices, :]

    return pts_3d_inside_img, pts_2d, inside_pts_indices


def get_lidar_on_image(l2c_object, pc_velo, size):
    """ Project LiDAR points to image """
    imgfov_pc_velo, all_pts_2d, fov_inds = remove_lidar_points_beyond_img(l2c_object,
        pc_velo, 0, 0, size[0], size[1], 1.5
    )

    return imgfov_pc_velo, all_pts_2d[fov_inds, :]


def display_lidar_on_image(l2c_object, pc_velo, image):
    img = image.copy()
    imgfov_pc_velo, pts_2d = get_lidar_on_image(l2c_object, pc_velo, (img.shape[1], img.shape[0]))

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(pts_2d.shape[0]):
        depth = imgfov_pc_velo[i, 0]

        # if depth is 2, then 510 / 2 will be  255, white color
        # 510 has been calculated according to the clip distance, usually clip distance is 2
        # therefore we get depth color in the range (0, 255)
        color = cmap[int(510.0 / depth), :]
        pt = (int(np.round(pts_2d[i, 0])), int(np.round(pts_2d[i, 1])))
        cv2.circle(img, pt, 2, color=tuple(color), thickness=-1)

    return img


def convert_bin_to_pcd(point_files):
    list_pcd = []
    size_float = 4

    for i in range(len(point_files)):
        file_to_open = point_files[i]
        file_to_save = str(point_files[i])[:-3] + "pcd"
        with open(file_to_open, "rb") as f:
            byte = f.read(size_float * 4)
            while byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)
    o3d.io.write_point_cloud(file_to_save, pcd)


#FusionUtilites

def build_fused_object(list_of_2d_objects, list_of_3d_objects, matches, image):
    "Input: Image with 3D Boxes already drawn"
    final_image = image.copy()
    list_of_fused_objects = []
    for match in matches:
        fused_object = fu.FusedObject(list_of_2d_objects[match[1]].bbox, list_of_3d_objects[match[0]].bbox3d,
                                   list_of_2d_objects[match[1]].category, list_of_3d_objects[match[0]].t,
                                   list_of_2d_objects[match[1]].confidence)
        cv2.putText(final_image, '{0:.2f} m'.format(fused_object.t[2]), (int(fused_object.bbox2d[0] + 15),int(fused_object.bbox2d[1] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 255), 1, cv2.LINE_AA)
        # cv2.putText(final_image, fused_object.class_, (int(fused_object.bbox2d[0]+15),int(fused_object.bbox2d[1]+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 255), 1, cv2.LINE_AA)
    return final_image, list_of_fused_objects


def rectContains(rect, pt, shrink_factor=0.0):

    x_min = rect[0]
    y_min = rect[1]
    width = rect[2]
    height = rect[3]

    center_x = x_min + width * 0.5
    center_y = y_min + height * 0.5

    new_width = width * (1 - shrink_factor)
    new_height = height * (1 - shrink_factor)

    x1 = int(center_x - new_width * 0.5)
    y1 = int(center_y - new_height * 0.5)
    x2 = int(center_x + new_width * 0.5)
    y2 = int(center_y + new_height * 0.5)

    return x1 < pt[0] < x2 and y1 < pt[1] < y2


def filter_outliers(distances):
    inliers = []
    mu  = st.mean(distances)
    std = st.stdev(distances)
    for x in distances:
        if abs(x-mu) < std:
            # This is an INLIER
            inliers.append(x)
    return inliers


def get_best_distance(distances, technique="closest"):
    if technique == "closest":
        return min(distances)
    elif technique =="average":
        return st.mean(distances)
    elif technique == "random":
        return random.choice(distances)
    else:
        return st.median(sorted(distances))


def lidar_camera_fusion(pts_3D, pts_2D, detections, image):
    img_bis = image.copy()
    pred_bboxes = detections[:, 1]
    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    distances = []
    for box in pred_bboxes:
        distances = []
        for i in range(pts_2D.shape[0]):
            depth = pts_3D[i, 0]
            if rectContains(box, pts_2D[i], 0.1):
                distances.append(depth)

                color = cmap[int(510.0 / depth), :]
                cv2.circle(img_bis, (int(np.round(pts_2D[i, 0])), int(np.round(pts_2D[i, 1]))),
                           2, color=tuple(color), thickness=-1, )

        h, w, _ = img_bis.shape
        if len(distances) > 2:
            distances = filter_outliers(distances)
            best_distance = get_best_distance(distances, technique="average")
            cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (int(box[0] * w), int(box[1] * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (int(box[0] * w), int(box[1] * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
        distances_to_keep = []

    return img_bis, distances


def associate(lidar_boxes, camera_boxes):
    """
    Associates LiDAR and camera boxes using a greedy algorithm based on IOU as the metric.

    Parameters:
    - lidar_boxes (list of np.array): List of bounding boxes from LiDAR detections.
    - camera_boxes (list of np.array): List of bounding boxes from camera detections.

    Returns:
    - matches (np.array): Array of matched box indices.
    - unmatched_lidar_boxes (np.array): Array of unmatched LiDAR boxes.
    - unmatched_camera_boxes (np.array): Array of unmatched camera boxes.
    """

    matches = []
    unmatched_lidar_boxes = []
    unmatched_camera_boxes = camera_boxes.copy()

    iou_matrix = np.zeros((len(lidar_boxes), len(camera_boxes)), dtype=np.float32)
    for i, lidar_box in enumerate(lidar_boxes):
        for j, camera_box in enumerate(camera_boxes):
            iou_matrix[i][j] = get_iou(lidar_box, camera_box)

    while iou_matrix.size > 0:
        max_iou_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        max_iou_value = iou_matrix[max_iou_idx]
        
        if max_iou_value < 0.4:
            break
        
        matches.append(max_iou_idx)
        iou_matrix[max_iou_idx[0], :] = -1  # Mark this row as used
        iou_matrix[:, max_iou_idx[1]] = -1  # Mark this column as used
        
        unmatched_camera_boxes = [box for idx, box in enumerate(camera_boxes) if idx != max_iou_idx[1]]

    unmatched_lidar_boxes = [lidar_boxes[i] for i in range(len(lidar_boxes)) if all(i != m[0] for m in matches)]

    return np.array(matches), np.array(unmatched_lidar_boxes), np.array(unmatched_camera_boxes)














