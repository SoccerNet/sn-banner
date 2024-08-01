# Description: Camera parameters filtering functions

import numpy as np
from scipy.signal import medfilt, savgol_filter
import cv2
from multiprocessing import Pool

from camera_calibration.No_Bells_Just_Whistles.sn_calibration.src.camera import (
    Camera,
    pan_tilt_roll_to_orientation,
    rotation_matrix_to_pan_tilt_roll,
)


def to_valid_cam_params(camParamsPerImage):
    """Converts the camera parameters per image to a valid format, i.e. a
    dictionary per image, by replacing non-complete camera parameters with a
    temporary and random complete camera parameter.
    """
    for param in camParamsPerImage:
        if type(param) == dict:
            tmpCompleteParam = param
            break
    isErroneousParams = np.array([type(x) != dict for x in camParamsPerImage])
    ErroneousParamsPos = np.where(isErroneousParams)[0]
    for pos in ErroneousParamsPos:
        camParamsPerImage[pos] = tmpCompleteParam
    return camParamsPerImage, isErroneousParams, ErroneousParamsPos


def camParamsPerImage_to_camParamsPerType(camParamsPerImage):
    camParamsPerType = {
        "pan_degrees": np.array([x["pan_degrees"] for x in camParamsPerImage]),
        "tilt_degrees": np.array([x["tilt_degrees"] for x in camParamsPerImage]),
        "roll_degrees": np.array([x["roll_degrees"] for x in camParamsPerImage]),
        "position_meters": np.array([x["position_meters"] for x in camParamsPerImage]),
        "x_focal_length": np.array([x["x_focal_length"] for x in camParamsPerImage]),
        "y_focal_length": np.array([x["y_focal_length"] for x in camParamsPerImage]),
        "principal_point": np.array([x["principal_point"] for x in camParamsPerImage]),
        "radial_distortion": np.array(
            [x["radial_distortion"] for x in camParamsPerImage]
        ),
        "tangential_distortion": np.array(
            [x["tangential_distortion"] for x in camParamsPerImage]
        ),
        "thin_prism_distortion": np.array(
            [x["thin_prism_distortion"] for x in camParamsPerImage]
        ),
    }
    return camParamsPerType


def camParamsPerType_to_camParamsPerImage(camParamsPerType):
    camParamsPerImage = [
        dict(
            zip(
                camParamsPerType.keys(),
                [camParamsPerType[key][i].tolist() for key in camParamsPerType.keys()],
            )
        )
        for i in range(
            len(camParamsPerType["pan_degrees"])  # length of the camera parameters
        )
    ]
    return camParamsPerImage


def linear_interpolation(camParamsPerType, isErroneousParams, ErroneousParamsPos):
    """Linear interpolation of erroneous camera parameters"""

    # length of the camera parameters, any key can be used
    length = len(camParamsPerType["pan_degrees"])
    # xp = positions of complete camera parameters next to non-complete camera
    # parameters
    xp = []
    if not isErroneousParams[0] and isErroneousParams[1]:
        xp.append(0)
    for i in range(1, length - 1):
        if not isErroneousParams[i] and (
            isErroneousParams[i - 1] or isErroneousParams[i + 1]
        ):
            xp.append(i)
    if not isErroneousParams[-1] and isErroneousParams[-2]:
        xp.append(length - 1)
    if len(xp) == 0:
        return camParamsPerType
    for key, value in camParamsPerType.items():
        if len(value.shape) == 1:
            camParamsPerType[key][ErroneousParamsPos] = np.interp(
                ErroneousParamsPos, xp, value[xp]
            )
        else:  # 2D array
            for i in range(value.shape[1]):
                camParamsPerType[key][ErroneousParamsPos, i] = np.interp(
                    ErroneousParamsPos, xp, value[xp, i]
                )
    return camParamsPerType


def outliers_remover(
    camParamsPerType, isErroneousParams, ErroneousParamsPos, windowLength=11
):
    """Removes outliers from camera parameters. Outliers are detected by
    comparing the absolute difference between the camera parameters and their
    median filtered version with the mean absolute difference. If the absolute
    difference is more than twice the mean absolute difference, the camera
    parameter is considered an outlier and is linearly interpolated.
    """

    camParamsLength = len(camParamsPerType["pan_degrees"])
    for _, paramValues2d in camParamsPerType.items():
        if len(paramValues2d.shape) == 1:
            paramValues2d = np.array([paramValues2d])
        else:
            paramValues2d = paramValues2d.T

        for paramValues1d in paramValues2d:
            median_filtered_param_values = medfilt(paramValues1d, windowLength)
            abs_diff_param_values = np.abs(paramValues1d - median_filtered_param_values)
            mean_abs_diff_param_values = np.mean(abs_diff_param_values)
            newErroneousParamsPos = np.where(
                abs_diff_param_values > mean_abs_diff_param_values * 2
            )[0]
            ErroneousParamsPos = np.union1d(ErroneousParamsPos, newErroneousParamsPos)
            isErroneousParams = np.zeros(camParamsLength, dtype=np.bool_)
            isErroneousParams[ErroneousParamsPos] = True

    camParamsPerType = linear_interpolation(
        camParamsPerType, isErroneousParams, ErroneousParamsPos
    )

    return camParamsPerType


def camParamsSmoothing(camParamsPerType, windowLength=17):
    for key in camParamsPerType.keys():
        camParamsPerType[key] = savgol_filter(
            camParamsPerType[key], windowLength, 2, axis=0
        )
    # clamp values of radial_distortion, tangential_distortion and thin_prism_distortion btw 0 and +inf
    for key in ["radial_distortion", "tangential_distortion", "thin_prism_distortion"]:
        camParamsPerType[key] = np.maximum(camParamsPerType[key], 0)
    return camParamsPerType


def smoothing_using_banner_corners(
    smoothedCamParamsPerImage,
    basicCamParamsPerImage,
    bannersObjPts,
    LeftCornerImgPtPerImage,
    RightCornerImgPtPerImage,
):
    # The function worker is very fast, no need to use more than 1 process
    with Pool(1) as p:
        res = p.starmap(
            smoothing_using_banner_corners_worker,
            zip(
                smoothedCamParamsPerImage,
                basicCamParamsPerImage,
                [bannersObjPts] * len(smoothedCamParamsPerImage),
                LeftCornerImgPtPerImage,
                RightCornerImgPtPerImage,
            ),
        )
    return res


def smoothing_using_banner_corners_worker(
    smoothedCamParams,
    basicCamParams,
    bannersObjPts,
    LeftCornerImgPt,
    RightCornerImgPt,
):
    cam = Camera()
    cam.from_json_parameters(smoothedCamParams)
    R, c, cameraMatrix = cam.rotation, cam.position, cam.calibration
    rvec = cv2.Rodrigues(R)[0]
    tvec = -R @ c.reshape(-1, 1)  # type: ignore
    cornerType = "no corner"
    if not np.any(np.isnan(LeftCornerImgPt)):
        cornerType = "left"
    elif not np.any(np.isnan(RightCornerImgPt)):
        cornerType = "right"

    if cornerType == "left":
        imgPtsMiddleBan = cv2.projectPoints(bannersObjPts["middle"], rvec, tvec, cameraMatrix, None)[0][:, 0, :]  # type: ignore
        diff = LeftCornerImgPt - imgPtsMiddleBan[0]
        imgPtsMiddleBan[1][0] = imgPtsMiddleBan[0][0]
        imgPtsMiddleBan[2][0] = imgPtsMiddleBan[3][0]
        imgPtsLeftBan = cv2.projectPoints(bannersObjPts["left"], rvec, tvec, cameraMatrix, None)[0][:, 0, :]  # type: ignore
        imgPtsLeftBan[1][0] = imgPtsLeftBan[0][0]
        imgPtsLeftBan[2][0] = imgPtsLeftBan[3][0]
        allObjPts = np.concatenate([bannersObjPts["middle"], bannersObjPts["left"]])
        allImgPts = np.concatenate([imgPtsMiddleBan, imgPtsLeftBan])
        allImgPts += diff
    elif cornerType == "right":
        imgPtsMiddleBan = cv2.projectPoints(bannersObjPts["middle"], rvec, tvec, cameraMatrix, None)[0][:, 0, :]  # type: ignore
        diff = RightCornerImgPt - imgPtsMiddleBan[3]
        imgPtsMiddleBan[1][0] = imgPtsMiddleBan[0][0]
        imgPtsMiddleBan[2][0] = imgPtsMiddleBan[3][0]
        imgPtsRightBan = cv2.projectPoints(bannersObjPts["right"], rvec, tvec, cameraMatrix, None)[0][:, 0, :]  # type: ignore
        imgPtsRightBan[1][0] = imgPtsRightBan[0][0]
        imgPtsRightBan[2][0] = imgPtsRightBan[3][0]
        allObjPts = np.concatenate([bannersObjPts["middle"], bannersObjPts["right"]])
        allImgPts = np.concatenate([imgPtsMiddleBan, imgPtsRightBan])
        allImgPts += diff

    if cornerType != "no corner":
        cameraMatrix = np.array(cameraMatrix, dtype=np.float64)
        rvec, tvec = cv2.solvePnPRefineLM(allObjPts, allImgPts, cameraMatrix, None, rvec, tvec)  # type: ignore
        rotation, _ = cv2.Rodrigues(rvec)
        position = -np.transpose(rotation) @ tvec.flatten()
        pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(rotation)
        if -np.pi / 2 > pan or pan > np.pi / 2:
            dpi = -np.sign(pan) * np.pi
            pan += dpi
            roll *= -1
            position[2] *= -1
        rotation = np.transpose(pan_tilt_roll_to_orientation(pan, tilt, roll))
        cam = Camera()
        cam.from_json_parameters(basicCamParams)
        cam.position = position
        cam.rotation = rotation
    else:
        cam = Camera()
        cam.from_json_parameters(basicCamParams)
    return cam.to_json_parameters()
