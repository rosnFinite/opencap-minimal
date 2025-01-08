import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

openPoseMarkerNames = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                       "LShoulder", "LElbow", "LWrist", "midHip", "RHip",
                       "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                       "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                       "LHeel", "RBigToe", "RSmallToe", "RHeel"]

# pose skeleton
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
    (1, 5), (5, 6), (6, 7),  # Left arm
    (1, 8), (8, 9), (9, 10), (10, 11),  # Right leg
    (8, 12), (12, 13), (13, 14),  # Left leg
    (0, 15), (0, 16), (15, 17), (16, 18),  # Eyes and ears
    (11, 22), (14, 21), (22, 23), (23, 24), (21, 19), (19, 20)  # Feet
]


# Function to calculate angle between two vectors
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure numerical stability
    return np.degrees(np.arccos(cos_theta))


def calculate_knee_agles(kp_data, r_hip_idx, r_knee_idx, r_ankle_idx, l_hip_idx, l_knee_idx, l_ankle_idx):
    # Calculate knee angles for all frames
    right_knee_angles = []
    left_knee_angles = []
    for frame_idx in range(kp_data.shape[1]):
        # Right knee
        right_hip = kp_data[r_hip_idx, frame_idx]
        right_knee = kp_data[r_knee_idx, frame_idx]
        right_ankle = kp_data[r_ankle_idx, frame_idx]
        right_femur_vector = right_knee - right_hip
        right_tibia_vector = right_ankle - right_knee
        right_knee_angles.append(calculate_angle(right_femur_vector, right_tibia_vector))

        # Left knee
        left_hip = kp_data[l_hip_idx, frame_idx]
        left_knee = kp_data[l_knee_idx, frame_idx]
        left_ankle = kp_data[l_ankle_idx, frame_idx]
        left_femur_vector = left_knee - left_hip
        left_tibia_vector = left_ankle - left_knee
        left_knee_angles.append(calculate_angle(left_femur_vector, left_tibia_vector))
    return right_knee_angles, left_knee_angles


def transform_kp_data(kp_data):
    # reshape
    transformed_kp = np.transpose(kp_data, (1, 2, 0))  # Shape becomes (25, 776, 3)
    transformed_kp[:, :, [1, 2]] = transformed_kp[:, :, [2, 1]]
    # transform negative y values to positive to allow for an upright standing skeleton visualizations
    transformed_kp[:, :, 2] = -transformed_kp[:, :, 2]
    return transformed_kp


def show_3d_pose(keypoints_pkl_path: str,
                      title: str = "3D Keypoint Visualization",
                      max_frames: int = 100):
    keypoints_3d = pd.read_pickle(keypoints_pkl_path)

    # limit number of frames -> visualization performance
    if keypoints_3d.shape[2] >= max_frames:
        keypoints_3d = keypoints_3d[:, :, :max_frames]

    transformed_kp = transform_kp_data(keypoints_3d)

    # some keypoints may have many outliers over the whole video e.g. foot
    # here we can exclude them to make the plot mor stable
    excluded_keypoints = ["LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    included_keypoints = [i for i, name in enumerate(openPoseMarkerNames) if name not in excluded_keypoints]

    # filter keypoints to visualize
    transformed_kp = transformed_kp[included_keypoints, :, :]
    filtered_markers = [openPoseMarkerNames[i] for i in included_keypoints]
    # Filter connections to exclude invalid indices
    connections = [
        (included_keypoints.index(i), included_keypoints.index(j))
        for i, j in edges
        if i in included_keypoints and j in included_keypoints
    ]

    fig = go.Figure()

    # Add the first frame (frame 0) as the initial scatter plot
    xs = transformed_kp[:, 0, 0]  # X-coordinates for frame 0
    ys = transformed_kp[:, 0, 1]  # Y-coordinates for frame 0
    zs = transformed_kp[:, 0, 2]  # Z-coordinates for frame 0

    fig.add_trace(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers+text',
        text=filtered_markers,
        textposition="top center",
        marker=dict(size=2, color="blue"),
        name="Frame 0"
    ))

    # Add skeleton connections for the first frame
    for connection in connections:
        i, j = connection
        fig.add_trace(go.Scatter3d(
            x=[xs[i], xs[j]],
            y=[ys[i], ys[j]],
            z=[zs[i], zs[j]],
            mode='lines',
            line=dict(color='black', width=2),
            name=f"Connection {i}-{j}"
        ))

    # Create frames for animation
    frames = []
    for frame_idx in range(transformed_kp.shape[1]):
        xs = transformed_kp[:, frame_idx, 0]
        ys = transformed_kp[:, frame_idx, 1]
        zs = transformed_kp[:, frame_idx, 2]

        # Frame data
        frame_data = [
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='markers+text',
                text=filtered_markers,
                textposition="top center",
                marker=dict(size=2, color="blue"),
            )
        ]

        # Add skeleton connections for this frame
        for connection in connections:
            i, j = connection
            frame_data.append(go.Scatter3d(
                x=[xs[i], xs[j]],
                y=[ys[i], ys[j]],
                z=[zs[i], zs[j]],
                mode='lines',
                line=dict(color='black', width=2),
            ))

        # Append frame
        frames.append(go.Frame(data=frame_data, name=f"Frame {frame_idx}"))

    fig.frames = frames

    # Add animation controls
    fig.update_layout(
        sliders=[
            {
                "steps": [
                    {
                        "args": [[f"Frame {i}"], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                        "label": f"{i}",
                        "method": "animate"
                    }
                    for i in range(transformed_kp.shape[1])
                ],
                "currentvalue": {"prefix": "Frame: "},
            }
        ],
    )

    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Z"),
            zaxis=dict(title="Y"),
            camera=dict(
                eye=dict(x=2, y=2, z=2)  # Adjust x, y, z to zoom out
            )
        ),
    )


    fig.show()


def show_knee_angles(keypoints_pkl_path: str, knee_side="L"):
    if knee_side not in ["L", "B", "R"]:
        raise ValueError("knee_side must be 'L' for left knee, 'R' for right knee or 'B' for both.")

    keypoints_3d = pd.read_pickle(keypoints_pkl_path)

    # indices of Hip, Knee and Ankle for L or R to calculate knee angle
    right_hip_idx = 9
    right_knee_idx = 10
    right_ankle_idx = 11

    left_hip_idx = 12
    left_knee_idx = 13
    left_ankle_idx = 14

    transformed_kp = transform_kp_data(keypoints_3d)
    right_knee_angles, left_knee_angles = calculate_knee_agles(transformed_kp,
                                                               r_hip_idx=right_hip_idx,
                                                               r_knee_idx=right_knee_idx,
                                                               r_ankle_idx=right_ankle_idx,
                                                               l_hip_idx=left_hip_idx,
                                                               l_knee_idx=left_knee_idx,
                                                               l_ankle_idx=left_ankle_idx)

    fig = go.Figure()
    if knee_side in ["R", "B"]:
        fig.add_trace(go.Scatter(
            x=list(range(len(right_knee_angles))),
            y=right_knee_angles,
            mode='lines',
            line=dict(color='blue', width=2),
            name="Right"
        ))
    if knee_side in ["L", "B"]:
        fig.add_trace(go.Scatter(
            x=list(range(len(left_knee_angles))),
            y=left_knee_angles,
            mode='lines',
            line=dict(color="red", width=2),
            name="Left"
        ))
    fig.update_layout(
        title="Knee Angle in Degrees",
        yaxis=dict(title="Knee Angle (Degrees)"),
        showlegend=True
    )
    fig.show()


def create_3d_visualization(keypoints_pckl_path: str,
                            startIndOpen: int,
                            startIndMMPose: int,
                            html_path: str = None,
                            title: str = "3D Pose and Knee Angle Visualization"):
    keypoints_3d = pd.read_pickle(keypoints_pckl_path)

    # synchronize with openpose results
    frame_offset = startIndOpen - startIndMMPose
    if frame_offset > 0:
        # OpenPose starts later; trim first `frame_offset` frames of the current keypoints
        keypoints_3d = keypoints_3d[:, :, frame_offset:]
    elif frame_offset < 0:
        # OpenPose starts earlier; pad the current keypoints with empty frames at the beginning
        padding = np.full((keypoints_3d.shape[0], keypoints_3d.shape[1], abs(frame_offset)), np.nan)
        keypoints_3d = np.concatenate((padding, keypoints_3d), axis=2)

    # indices of Hip, Knee and Ankle for L or R to calculate knee angle
    right_hip_idx = 9
    right_knee_idx = 10
    right_ankle_idx = 11

    left_hip_idx = 12
    left_knee_idx = 13
    left_ankle_idx = 14

    transformed_kp = transform_kp_data(keypoints_3d)

    # some keypoints may have many outliers over the whole video e.g. foot
    # here we can exclude them to make the plot mor stable
    excluded_keypoints = ["LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    included_keypoints = [i for i, name in enumerate(openPoseMarkerNames) if name not in excluded_keypoints]


    # filter keypoints to visualize
    transformed_kp = transformed_kp[included_keypoints, :, :]
    filtered_markers = [openPoseMarkerNames[i] for i in included_keypoints]
    # Filter connections to exclude invalid indices
    connections = [
        (included_keypoints.index(i), included_keypoints.index(j))
        for i, j in edges
        if i in included_keypoints and j in included_keypoints
    ]

    right_knee_angles, left_knee_angles = calculate_knee_agles(transformed_kp,
                                                               r_hip_idx=right_hip_idx,
                                                               r_knee_idx=right_knee_idx,
                                                               r_ankle_idx=right_ankle_idx,
                                                               l_hip_idx=left_hip_idx,
                                                               l_knee_idx=left_knee_idx,
                                                               l_ankle_idx=left_ankle_idx)

    # Create the initial plots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scatter3d", "rowspan": 2}, {"type": "scatter"}],
            [None, {"type": "scatter"}]
        ],
        row_heights=[0.5, 0.5],  # Each knee angle plot takes half of the right column
        column_widths=[0.5, 0.5],  # Equal split between left and right halves
        horizontal_spacing=0.05,  # Space between columns
        vertical_spacing=0.05,    # Space between rows
        subplot_titles=("3D Pose", "Right Knee Angle", "Left Knee Angle")
    )

    # Add the first frame (frame 0) as the initial scatter plot
    xs = transformed_kp[:, 0, 0]  # X-coordinates for frame 0
    ys = transformed_kp[:, 0, 1]  # Y-coordinates for frame 0
    zs = transformed_kp[:, 0, 2]  # Z-coordinates for frame 0

    fig.add_trace(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers+text',
        text=filtered_markers,
        textposition="top center",
        marker=dict(size=2, color="blue"),
        name="Frame 0"
    ), row=1, col=1)

    # Add skeleton connections for the first frame
    for connection in connections:
        i, j = connection
        fig.add_trace(go.Scatter3d(
            x=[xs[i], xs[j]],
            y=[ys[i], ys[j]],
            z=[zs[i], zs[j]],
            mode='lines',
            line=dict(color='black', width=2),
            name=f"Connection {i}-{j}"
        ), row=1, col=1)

    # Add angle plot
    fig.add_trace(go.Scatter(
        x=list(range(len(right_knee_angles))),
        y=right_knee_angles,
        mode='lines',
        line=dict(color='blue', width=2),
        name="Knee Angle"
    ), row=1, col=2)

    fig.add_trace(
        go.Scatter(
            x=list(range(len(left_knee_angles))),
            y=left_knee_angles,
            mode="lines",
            line=dict(color="green", width=2),
            name="Left Knee Angle"
        ),
        row=2, col=2
    )

    # Add markers for the current frame on angle plots
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[right_knee_angles[0]],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Current Frame (Right Knee)"
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[left_knee_angles[0]],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Current Frame (Left Knee)"
        ),
        row=2, col=2
    )

    # Create frames for animation
    frames = []
    for frame_idx in range(transformed_kp.shape[1]):
        xs = transformed_kp[:, frame_idx, 0]
        ys = transformed_kp[:, frame_idx, 1]
        zs = transformed_kp[:, frame_idx, 2]

        # Frame data
        frame_data = [
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='markers+text',
                text=filtered_markers,
                textposition="top center",
                marker=dict(size=2, color="blue"),
            )
        ]

        # Add skeleton connections for this frame
        for connection in connections:
            i, j = connection
            frame_data.append(go.Scatter3d(
                x=[xs[i], xs[j]],
                y=[ys[i], ys[j]],
                z=[zs[i], zs[j]],
                mode='lines',
                line=dict(color='black', width=2),
            ))

        # Add static knee angle line plots
        frame_data.append(go.Scatter(
            x=list(range(len(right_knee_angles))),
            y=right_knee_angles,
            mode="lines",
            line=dict(color="blue", width=2),
            showlegend=False
        ))
        frame_data.append(go.Scatter(
            x=list(range(len(left_knee_angles))),
            y=left_knee_angles,
            mode="lines",
            line=dict(color="green", width=2),
            showlegend=False
        ))

        # Add moving markers for current frame
        frame_data.append(go.Scatter(
            x=[frame_idx],
            y=[right_knee_angles[frame_idx]],
            mode="markers",
            marker=dict(size=10, color="red"),
            showlegend=False
        ))
        frame_data.append(go.Scatter(
            x=[frame_idx],
            y=[left_knee_angles[frame_idx]],
            mode="markers",
            marker=dict(size=10, color="red"),
            showlegend=False
        ))

        # Append frame
        frames.append(go.Frame(data=frame_data, name=f"Frame {frame_idx}"))

    fig.frames = frames

    # Add animation controls
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "type": "buttons",
                "showactive": False
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [[f"Frame {i}"], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                        "label": f"{i}",
                        "method": "animate"
                    }
                    for i in range(transformed_kp.shape[1])
                ],
                "currentvalue": {"prefix": "Frame: "},
            }
        ],
    )

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Z"),
            zaxis=dict(title="Y"),
            camera=dict(
                eye=dict(x=2, y=2, z=2)  # Adjust x, y, z to zoom out
            )
        ),
        yaxis=dict(title="Right Knee Angle (Degrees)"),
        yaxis2=dict(title="Left Knee Angle (Degrees)"),
        showlegend=False
    )

    # Show the animation
    fig.show()

    if html_path is not None:
        fig.write_html(html_path)


if __name__ == "__main__":
    create_3d_visualization(keypoints_pckl_path="",
                            html_path="",
                            startIndOpen=9,
                            startIndMMPose=0)