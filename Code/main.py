import cv2
import os
import random
import glob
import numpy as np
import open3d as o3d
#import Utils as ut
import Fusion as Fu
import YoloDetector as yd

def feature_level_fusion(root_dir, display_image=True, save_image=True):
    imgs, pts, labels, calibs = Fu.load_data(root_dir)

    if save_image:
        out_dir = os.path.join(root_dir, "output/images")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    weights = os.path.join(root_dir, "model/yolov4/yolov4.weights")
    config = os.path.join(root_dir, "model/yolov4/yolov4.cfg")
    names = os.path.join(root_dir, "model/yolov4/coco.names")

    detector = yd.Detector(0.4)
    detector.load_model(weights, config, names)

    """PIPELINE STARTS FROM HERE"""

    for img_path in imgs:
        # load the image
        image = cv2.imread(img_path)
        image_name = os.path.basename(img_path)  # Get the base name of the image file

        # create LiDAR2Camera object
        lidar2cam = Fu.LiDAR2Camera(calibs[imgs.index(img_path)])

        # 1 - Run 2D object detection on image
        detections, yolo_detections = detector.detect(image, draw_bboxes=True, display_labels=True)

        # load lidar points and project them inside 2d detection
        point_cloud = np.asarray(o3d.io.read_point_cloud(pts[imgs.index(img_path)]).points)
        pts_3D, pts_2D = Fu.get_lidar_on_image(lidar2cam, point_cloud, (image.shape[1], image.shape[0]))
        lidar_pts_img, _ = Fu.lidar_camera_fusion(pts_3D, pts_2D, detections, image)

        # Build a 2D Object
        list_of_2d_objects = Fu.fill_2D_obstacles(detections)

        # Build a 3D Object (from labels)
        list_of_3d_objects = Fu.read_label(labels[imgs.index(img_path)])

        # Get the LiDAR Boxes in the Image in 2D and 3D
        lidar_2d, lidar_3d = Fu.get_image_with_bboxes(lidar2cam, lidar_pts_img, list_of_3d_objects)

        # Associate the LiDAR boxes and the Camera Boxes
        lidar_boxes = [obs.bbox2d for obs in list_of_3d_objects]  # Simply get the boxes
        camera_boxes = [np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]]) for box in detections[:, 1]]
        matches, unmatched_lidar_boxes, unmatched_camera_boxes = Fu.associate(lidar_boxes, camera_boxes)

        # Build a Fused Object
        final_image, _ = Fu.build_fused_object(list_of_2d_objects, list_of_3d_objects, matches, lidar_2d)

        # draw yolo detections on top to fused results
        final_image = yd.draw_yolo_detections(final_image, detections)

        if display_image:
            cv2.imshow("lidar_3d", lidar_3d)
            cv2.imshow("lidar_2d", lidar_2d)
            cv2.imshow("yolo_detections", yolo_detections)
            cv2.imshow("lidar_pts_img", lidar_pts_img)
            cv2.imshow("fused_result", final_image)
            cv2.waitKey(0)
        
        if save_image:
            output_filepath = os.path.join(out_dir, image_name)
            cv2.imwrite(output_filepath, final_image)

    return final_image

if __name__ == "__main__":
    root_dir = "../Data/"
    feature_level_fusion(root_dir, display_image=True, save_image=True)