import yaml
import json
import os
import socket
import requests
import shutil
from . import utilsDataman
import pickle
import glob
import subprocess
import time

import numpy as np
import pandas as pd
from scipy import signal

API_URL = None
API_TOKEN = None


# %% Rest of utils

def getDataDirectory(isDocker=False):
    computername = socket.gethostname()
    # Paths to OpenPose folder for local testing.
    if computername == 'SUHLRICHHPLDESK':
        dataDir = 'C:/Users/scott.uhlrich/MyDrive/mobilecap/'
    elif computername == "LAPTOP-7EDI4Q8Q":
        dataDir = 'C:\MyDriveSym/mobilecap/'
    elif computername == 'DESKTOP-0UPR1OH':
        dataDir = 'C:/Users/antoi/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'HPL1':
        dataDir = 'C:/Users/opencap/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'DESKTOP-GUEOBL2':
        dataDir = 'C:/Users/opencap/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'DESKTOP-L9OQ0MS':
        dataDir = 'C:/Users/antoi/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'clarkadmin-MS-7996':
        dataDir = '/home/clarkadmin/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'DESKTOP-NJMGEBG':
        dataDir = 'C:/Users/opencap/Documents/MyRepositories/mobilecap_data/'
    elif isDocker:
        dataDir = os.getcwd()
    else:
        dataDir = os.getcwd()
    return dataDir


def getOpenPoseDirectory(isDocker=False):
    computername = os.environ.get('COMPUTERNAME', None)

    # Paths to OpenPose folder for local testing.
    if computername == "DESKTOP-0UPR1OH":
        openPoseDirectory = "C:/Software/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif computername == "HPL1":
        openPoseDirectory = "C:/Users/opencap/Documents/MySoftware/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif computername == "DESKTOP-GUEOBL2":
        openPoseDirectory = "C:/Software/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif computername == "DESKTOP-L9OQ0MS":
        openPoseDirectory = "C:/Software/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif isDocker:
        openPoseDirectory = "docker"
    elif computername == 'SUHLRICHHPLDESK':
        openPoseDirectory = "C:/openpose/"
    elif computername == "LAPTOP-7EDI4Q8Q":
        openPoseDirectory = "C:/openpose/"
    elif computername == "DESKTOP-NJMGEBG":
        openPoseDirectory = "C:/openpose/"
    else:
        openPoseDirectory = "C:/openpose/"
    return openPoseDirectory


def getMMposeDirectory(isDocker=False):
    computername = socket.gethostname()

    # Paths to OpenPose folder for local testing.
    if computername == "clarkadmin-MS-7996":
        mmposeDirectory = "/home/clarkadmin/Documents/MyRepositories/MoVi_analysis/model_ckpts"
    else:
        mmposeDirectory = ''
    return mmposeDirectory


def loadCameraParameters(filename):
    open_file = open(filename, "rb")
    cameraParams = pickle.load(open_file)

    open_file.close()
    return cameraParams


def importMetadata(filePath):
    myYamlFile = open(filePath)
    parsedYamlFile = yaml.load(myYamlFile, Loader=yaml.FullLoader)

    return parsedYamlFile


def numpy2TRC(f, data, headers, fc=50.0, t_start=0.0, units="m"):
    header_mapping = {}
    for count, header in enumerate(headers):
        header_mapping[count + 1] = header

        # Line 1.
    f.write('PathFileType  4\t(X/Y/Z) %s\n' % os.getcwd())

    # Line 2.
    f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\t'
            'Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')

    num_frames = data.shape[0]
    num_markers = len(header_mapping.keys())

    # Line 3.
    f.write('%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n' % (
        fc, fc, num_frames,
        num_markers, units, fc,
        1, num_frames))

    # Line 4.
    f.write("Frame#\tTime\t")
    for key in sorted(header_mapping.keys()):
        f.write("%s\t\t\t" % format(header_mapping[key]))

    # Line 5.
    f.write("\n\t\t")
    for imark in np.arange(num_markers) + 1:
        f.write('X%i\tY%s\tZ%s\t' % (imark, imark, imark))
    f.write('\n')

    # Line 6.
    f.write('\n')

    for frame in range(data.shape[0]):
        f.write("{}\t{:.8f}\t".format(frame + 1, (frame) / fc + t_start))  # opensim frame labeling is 1 indexed

        for key in sorted(header_mapping.keys()):
            f.write("{:.5f}\t{:.5f}\t{:.5f}\t".format(data[frame, 0 + (key - 1) * 3], data[frame, 1 + (key - 1) * 3],
                                                      data[frame, 2 + (key - 1) * 3]))
        f.write("\n")


def TRC2numpy(pathFile, markers, rotation=None):
    # rotation is a dict, eg. {'y':90} with axis, angle for rotation

    trc_file = utilsDataman.TRCFile(pathFile)
    time = trc_file.time
    num_frames = time.shape[0]
    data = np.zeros((num_frames, len(markers) * 3))

    if rotation != None:
        for axis, angle in rotation.items():
            trc_file.rotate(axis, angle)
    for count, marker in enumerate(markers):
        data[:, 3 * count:3 * count + 3] = trc_file.marker(marker)
    this_dat = np.empty((num_frames, 1))
    this_dat[:, 0] = time
    data_out = np.concatenate((this_dat, data), axis=1)

    return data_out


def getOpenPoseMarkerNames():
    markerNames = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                   "LShoulder", "LElbow", "LWrist", "midHip", "RHip",
                   "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                   "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                   "LHeel", "RBigToe", "RSmallToe", "RHeel"]

    return markerNames


def getOpenPoseFaceMarkers():
    faceMarkerNames = ['Nose', 'REye', 'LEye', 'REar', 'LEar']
    markerNames = getOpenPoseMarkerNames()
    idxFaceMarkers = [markerNames.index(i) for i in faceMarkerNames]

    return faceMarkerNames, idxFaceMarkers


def getMMposeMarkerNames():
    markerNames = [
        # Body keypoints (17 total)
        "Nose", "LEye", "REye", "LEar", "REar",
        "LShoulder", "RShoulder", "LElbow", "RElbow",
        "LWrist", "RWrist", "LHip", "RHip",
        "LKnee", "RKnee", "LAnkle", "RAnkle",

        # Feet keypoints (6 total)
        "LBigToe", "LSmallToe", "LHeel",
        "RBigToe", "RSmallToe", "RHeel",

        # Face keypoints (68 total)
        "Contour_0", "Contour_1", "Contour_2", "Contour_3", "Contour_4",
        "Contour_5", "Contour_6", "Contour_7", "Contour_8", "Contour_9",
        "Contour_10", "Contour_11", "Contour_12", "Contour_13", "Contour_14",
        "Contour_15", "Contour_16",
        "REyebrow_0", "REyebrow_1", "REyebrow_2", "REyebrow_3", "REyebrow_4",
        "LEyebrow_0", "LEyebrow_1", "LEyebrow_2", "LEyebrow_3", "LEyebrow_4",
        "Nose_0", "Nose_1", "Nose_2", "Nose_3", "Nose_4", "Nose_5", "Nose_6", "Nose_7", "Nose_8",
        "REye_0", "REye_1", "REye_2", "REye_3", "REye_4", "REye_5",
        "LEye_0", "LEye_1", "LEye_2", "LEye_3", "LEye_4", "LEye_5",
        "Lip_0", "Lip_1", "Lip_2", "Lip_3", "Lip_4", "Lip_5", "Lip_6",
        "Lip_7", "Lip_8", "Lip_9", "Lip_10", "Lip_11",
        "Mouth_0", "Mouth_1", "Mouth_2", "Mouth_3", "Mouth_4", "Mouth_5",
        "Mouth_6", "Mouth_7",

        # Left hand keypoints (21 total)
        "LHand_0", "LHand_1", "LHand_2", "LHand_3", "LHand_4",
        "LHand_5", "LHand_6", "LHand_7", "LHand_8", "LHand_9",
        "LHand_10", "LHand_11", "LHand_12", "LHand_13", "LHand_14",
        "LHand_15", "LHand_16", "LHand_17", "LHand_18", "LHand_19", "LHand_20",

        # Right hand keypoints (21 total)
        "RHand_0", "RHand_1", "RHand_2", "RHand_3", "RHand_4",
        "RHand_5", "RHand_6", "RHand_7", "RHand_8", "RHand_9",
        "RHand_10", "RHand_11", "RHand_12", "RHand_13", "RHand_14",
        "RHand_15", "RHand_16", "RHand_17", "RHand_18", "RHand_19", "RHand_20",

        # Additional keypoints (6 total)
        "Neck", "MidHip", "NoseTip", "LeftPupil", "RightPupil"
    ]

    return markerNames


def rewriteVideos(inputPath, startFrame, nFrames, frameRate, outputDir=None,
                  imageScaleFactor=.5, outputFileName=None):
    inputDir, vidName = os.path.split(inputPath)
    vidName, vidExt = os.path.splitext(vidName)

    if outputFileName is None:
        outputFileName = vidName + '_sync' + vidExt
    if outputDir is not None:
        outputFullPath = os.path.join(outputDir, outputFileName)
    else:
        outputFullPath = os.path.join(inputDir, outputFileName)

    imageScaleArg = ''  # None if want to keep image size the same
    maintainQualityArg = '-acodec copy -vcodec copy'
    if imageScaleFactor is not None:
        imageScaleArg = '-vf scale=iw/{:.0f}:-1'.format(1 / imageScaleFactor)
        maintainQualityArg = ''

    startTime = startFrame / frameRate

    # We need to replace double space to single space for split to work
    # That's a bit hacky but works for now. (TODO)
    ffmpegCmd = "ffmpeg -loglevel error -y -ss {:.3f} -i {} {} -vframes {:.0f} {} {}".format(
        startTime, inputPath, maintainQualityArg,
        nFrames, imageScaleArg, outputFullPath).rstrip().replace("  ", " ")

    subprocess.run(ffmpegCmd.split(" "))

    return


# %%  Found here: https://github.com/chrisdembia/perimysium/ => thanks Chris
def storage2numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' tmpVisualizationOutputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
                         skip_header=skip_header)

    return data


def storage2df(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])

    return out


def getIK(storage_file, joints, degrees=False):
    # Extract data
    data = storage2numpy(storage_file)
    Qs = pd.DataFrame(data=data['time'], columns=['time'])
    for count, joint in enumerate(joints):
        if ((joint == 'pelvis_tx') or (joint == 'pelvis_ty') or
                (joint == 'pelvis_tz')):
            Qs.insert(count + 1, joint, data[joint])
        else:
            if degrees == True:
                Qs.insert(count + 1, joint, data[joint])
            else:
                Qs.insert(count + 1, joint, data[joint] * np.pi / 180)

                # Filter data
    fs = 1 / np.mean(np.diff(Qs['time']))
    fc = 6  # Cut-off frequency of the filter
    order = 4
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(order / 2, w, 'low')
    output = signal.filtfilt(b, a, Qs.loc[:, Qs.columns != 'time'], axis=0,
                             padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
    output = pd.DataFrame(data=output, columns=joints)
    QsFilt = pd.concat([pd.DataFrame(data=data['time'], columns=['time']),
                        output], axis=1)

    return Qs, QsFilt


# %% Markers for augmenters.
def getOpenPoseMarkers_fullBody():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study",
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study",
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study",
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
                        "r_sh1_study", "r_sh2_study", "r_sh3_study",
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers


def getMMposeMarkers_fullBody():
    # Here we replace RSmallToe_mmpose and LSmallToe_mmpose by RSmallToe and
    # LSmallToe, since this is how they are named in the triangulation.
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RElbow", "LElbow", "RWrist", "LWrist"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study",
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study",
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study",
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
                        "r_sh1_study", "r_sh2_study", "r_sh3_study",
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers


def getOpenPoseMarkers_lowerExtremity():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study",
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study",
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study",
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
                        "r_sh1_study", "r_sh2_study", "r_sh3_study",
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers


# Different order of markers compared to getOpenPoseMarkers_lowerExtremity
def getOpenPoseMarkers_lowerExtremity2():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]

    response_markers = [
        'r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study',
        'L.PSIS_study', 'r_knee_study', 'r_mknee_study',
        'r_ankle_study', 'r_mankle_study', 'r_toe_study',
        'r_5meta_study', 'r_calc_study', 'L_knee_study',
        'L_mknee_study', 'L_ankle_study', 'L_mankle_study',
        'L_toe_study', 'L_calc_study', 'L_5meta_study',
        'r_shoulder_study', 'L_shoulder_study', 'C7_study',
        'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
        'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
        'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 'L_sh1_study',
        'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']

    return feature_markers, response_markers


def getMMposeMarkers_lowerExtremity():
    # Here we replace RSmallToe_mmpose and LSmallToe_mmpose by RSmallToe and
    # LSmallToe, since this is how they are named in the triangulation.
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study",
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study",
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study",
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
                        "r_sh1_study", "r_sh2_study", "r_sh3_study",
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers


def getMarkers_upperExtremity_pelvis():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RElbow", "LElbow",
        "RWrist", "LWrist"]

    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers


def getMarkers_upperExtremity_noPelvis():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist",
        "LWrist"]

    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers


# Different order of markers compared to getMarkers_upperExtremity_noPelvis.
def getMarkers_upperExtremity_noPelvis2():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist",
        "LWrist"]

    response_markers = ["r_lelbow_study", "r_melbow_study", "r_lwrist_study",
                        "r_mwrist_study", "L_lelbow_study", "L_melbow_study",
                        "L_lwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers


def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def getVideoExtension(pathFileWithoutExtension):
    pathVideoDir = os.path.split(pathFileWithoutExtension)[0]
    videoName = os.path.split(pathFileWithoutExtension)[1]
    for file in os.listdir(pathVideoDir):
        if videoName == file.rsplit('.', 1)[0]:
            extension = '.' + file.rsplit('.', 1)[1]

    return extension


# check how much time has passed since last status check
def checkTime(t, minutesElapsed=30):
    t2 = time.localtime()
    return (t2.tm_hour - t.tm_hour) * 3600 + (t2.tm_min - t.tm_min) * 60 + (t2.tm_sec - t.tm_sec) >= minutesElapsed * 60


# check for trials with certain status
def checkForTrialsWithStatus(status, hours=9999999, relativeTime='newer'):
    # get trials with statusOld
    params = {'status': status,
              'hoursSinceUpdate': hours,
              'justNumber': 1,
              'relativeTime': relativeTime}

    r = requests.get(API_URL + "trials/get_trials_with_status/", params=params,
                     headers={"Authorization": "Token {}".format(API_TOKEN)}).json()

    return r['nTrials']


# send status email
def sendStatusEmail(message=None, subject=None):
    return ('No email info or wrong email info in env file.')


def checkResourceUsage(stop_machine_and_email=True):
    import psutil

    resourceUsage = {}

    memory_info = psutil.virtual_memory()
    resourceUsage['memory_gb'] = memory_info.used / (1024 ** 3)
    resourceUsage['memory_perc'] = memory_info.percent

    # Get the disk usage information of the root directory
    disk_usage = psutil.disk_usage('/')

    # Get the percentage of disk usage
    resourceUsage['disk_gb'] = disk_usage.used / (1024 ** 3)
    resourceUsage['disk_perc'] = disk_usage.percent

    if stop_machine_and_email and resourceUsage['disk_perc'] > 95:
        message = "Disc is full on an OpenCap backend machine. It has been stopped. Data: " \
                  + json.dumps(resourceUsage)
        sendStatusEmail(message=message)

        raise Exception('Not enough available disc space. Stopped.')

    return resourceUsage


def checkCudaTF():
    import tensorflow as tf

    if tf.config.list_physical_devices('GPU'):
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Found {len(gpus)} GPU(s).")
        for gpu in gpus:
            print(f"GPU: {gpu.name}")
    else:
        message = "Cuda check failed on an OpenCap backend machine. It has been stopped."
        sendStatusEmail(message=message)
        raise Exception("No GPU detected. Exiting.")


# %% Some functions for loading subject data

def getSubjectNumber(subjectName):
    subjects = requests.get(API_URL + "subjects/",
                            headers={"Authorization": "Token {}".format(API_TOKEN)}).json()
    sNum = [s['id'] for s in subjects if s['name'] == subjectName]
    if len(sNum) > 1:
        print(len(sNum) + ' subjects with the name ' + subjectName + '. Will use the first one.')
    elif len(sNum) == 0:
        raise Exception('no subject found with this name.')

    return sNum[0]


def getUserSessions():
    sessionJson = requests.get(API_URL + "sessions/valid/",
                               headers={"Authorization": "Token {}".format(API_TOKEN)}).json()
    return sessionJson


def getSubjectSessions(subjectName):
    sessions = getUserSessions()
    subNum = getSubjectNumber(subjectName)
    sessions2 = [s for s in sessions if (s['subject'] == subNum)]

    return sessions2


def getTrialNames(session):
    trialNames = [t['name'] for t in session['trials']]
    return trialNames


def findSessionWithTrials(subjectTrialNames, trialNames):
    hasTrials = []
    for trials in trialNames:
        hasTrials.append(None)
        for i, sTrials in enumerate(subjectTrialNames):
            if all(elem in sTrials for elem in trials):
                hasTrials[-1] = i
                break

    return hasTrials


def get_entry_with_largest_number(trialList):
    max_entry = None
    max_number = float('-inf')

    for entry in trialList:
        # Extract the number from the string
        try:
            number = int(entry.split('_')[-1])
            if number > max_number:
                max_number = number
                max_entry = entry
        except ValueError:
            continue

    return max_entry


# Get local client info and update

def getCommitHash():
    """Get the git commit hash stored in the environment variable
    GIT_COMMIT_HASH. This is assumed to be set in the Docker build
    step. If not set, returns Null (default value for os.getenv())
    """
    return os.getenv('GIT_COMMIT_HASH')


def getHostname():
    """Get the hostname. For a docker container, this is the container ID."""
    return socket.gethostname()


def postLocalClientInfo(trial_url):
    """Given a trial_url, updates the Trial fields for 
    'git_commit' and 'hostname'.
    """
    data = {
        "git_commit": getCommitHash(),
        "hostname": getHostname()
    }
    r = requests.patch(trial_url, data=data,
                       headers={"Authorization": "Token {}".format(API_TOKEN)})

    return r


def postProcessedDuration(trial_url, duration):
    """Given a trial_url and duration (formed from difference in datetime
    objects), updates the Trial field for 'processed_duration'.
    """
    data = {
        "processed_duration": duration
    }
    r = requests.patch(trial_url, data=data,
                       headers={"Authorization": "Token {}".format(API_TOKEN)})

    return r
