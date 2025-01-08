import argparse
import os
import pickle
import shutil

from mmpose.apis import MMPoseInferencer

# Argumente parsen
parser = argparse.ArgumentParser(description="Run MMPose inference")
parser.add_argument("--video", type=str, required=True, help="Path to video file")
parser.add_argument("--poseEstimation", type=str, required=True, help="Storage path of the estimated keypoints")
parser.add_argument("--boundingBox", required=True, help="Storage path of the detected bounding boxes")
parser.add_argument("--videoWithKeypoints", required=True, help="Storage path of the resulting visualization video")
parser.add_argument("--modelCfg", required=True, help="Path to the MMpose model configuration file")
parser.add_argument("--modelCkp", required=True, help="URL to the MMpose model checkpoint")
parser.add_argument("--bboxThr", required=True, type=float, help="Threshold for person bounding box detection.")
args = parser.parse_args()


def delete_all_files_in_folder(folder_path):
    try:
        # Check if the folder exists
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Iterate through all files in the folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                # Check if it is a file (not a folder)
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Delete the file
                    print(f"Deleted: {file_path}")
        else:
            print(f"The folder '{folder_path}' does not exist or is not a directory.")
    except Exception as e:
        print(f"An error occurred: {e}")


def copy_file_to_another_folder(source_file_path, destination_file_path):
    try:
        # Copy the file to the new location with a new name
        shutil.copy(source_file_path, destination_file_path)
        print(f"File copied successfully to '{destination_file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    temp_video_folder = "./mmpose/tmpVisualizationOutputs"

    inferencer = MMPoseInferencer(pose2d=args.modelCfg,
                                  pose2d_weights=args.modelCkp)

    result_generator = inferencer(args.video, vis_out_dir=temp_video_folder, show_progress=True, bbox_thr=args.bboxThr)
    result = [result for result in result_generator]

    # move the tmp_video to the correct trial folder for opencap
    copy_file_to_another_folder(source_file_path=temp_video_folder + os.listdir(temp_video_folder)[0],
                                destination_file_path=args.videoWithKeypoints)

    # transform into separate objects for keypoints and bounding boxes
    new_kp_frames = []
    new_bb_frames = []
    for frame in result:
        predictions = frame["predictions"]

        kp_data = []
        bbox_data = []

        for person in predictions[0]:
            kp_data.append({"keypoint_scores": person["keypoint_scores"], "keypoints": person["keypoints"]})
            bbox_data.append({"bbox_score": person["bbox_score"], "bbox": person["bbox"]})

        new_kp_frames.append(kp_data)
        new_bb_frames.append(bbox_data)

    with open(args.poseEstimation, 'wb') as f:
        pickle.dump(new_kp_frames, f)

    with open(args.boundingBox, 'wb') as f:
        pickle.dump(new_bb_frames, f)

    # clear temporary video visualization folder
    delete_all_files_in_folder(temp_video_folder)
