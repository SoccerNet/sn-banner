from camera_calibration.No_Bells_Just_Whistles.sn_calibration.src.camera import (
    Camera,
)
import numpy as np
from tqdm import tqdm
import cv2
from multiprocessing import Pool
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from copy import deepcopy

# disable messages about Exception ignored in multiprocessing.pool
import warnings

# warnings.filterwarnings("ignore", category=RuntimeWarning)


def compute_banner_model_params_worker(i):
    cam = Camera()
    cam.from_json_parameters(camParamsPerImage[i])
    binMask = (
        cv2.imread(maskPath + str(i).zfill(6) + ".png", cv2.IMREAD_GRAYSCALE)
        .__gt__(0)  # type: ignore
        .astype(np.uint8)
    )
    firstNonZeroIdx = np.argmax(binMask, axis=0)
    colsWithNonZero = np.where(firstNonZeroIdx != 0)[0]
    flipped = cv2.flip(binMask, 0)
    flippedLastNonZeroIdx = np.argmax(flipped, axis=0)
    # colsWithNonZero is the set of columns that have two non-zero pixels
    # and the non-zero pixels are not at the top or bottom of the image
    colsWithNonZero = np.intersect1d(
        colsWithNonZero, np.where(flippedLastNonZeroIdx != 0)[0]
    )
    firstNonZeroIdx = firstNonZeroIdx[colsWithNonZero]
    flippedLastNonZeroIdx = flippedLastNonZeroIdx[colsWithNonZero]
    lastNonZeroIdx = imgHeight - flippedLastNonZeroIdx - 1
    imgPts = np.array([[x, y] for x, y in zip(colsWithNonZero, lastNonZeroIdx)])
    objPts = np.array(
        [cam.unproject_point_on_planeZ0(p, undistort=False) for p in imgPts]
    )
    objPts[:, 2] = -1
    R, c, cameraMatrix = cam.rotation, cam.position, cam.calibration
    rvec = cv2.Rodrigues(R)[0]
    tvec = -R @ c.reshape(-1, 1)  # type: ignore
    z1m = cv2.projectPoints(objPts, rvec, tvec, cameraMatrix, np.zeros(4))[0][:, 0, 1]
    realZ = (lastNonZeroIdx - firstNonZeroIdx) / (lastNonZeroIdx - z1m)
    bannerHeight = realZ.mean()

    leftBannerDist, middleBannerDist, rightBannerDist = np.nan, np.nan, np.nan

    objPtsX = objPts[:, 0]
    objPtsY = objPts[:, 1]

    # If a banner is not detected, the mean will be nan. It warns about it, but it is not a problem.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        leftBannerDistCond = np.bitwise_and(
            np.bitwise_and(objPtsX < -105 / 2, objPtsY > -68 / 2), objPtsY < 68 / 2
        )
        leftBannerDist = objPts[leftBannerDistCond][:, 0].mean()
        rightBannerDistCond = np.bitwise_and(
            np.bitwise_and(objPtsX > 105 / 2, objPtsY > -68 / 2), objPtsY < 68 / 2
        )
        rightBannerDist = objPts[rightBannerDistCond][:, 0].mean()
        middleBannerDistCond = np.bitwise_and(
            np.bitwise_and(objPtsX > -105 / 2, objPtsX < 105 / 2), objPtsY < -68 / 2
        )
        middleBannerDist = objPts[middleBannerDistCond][:, 1].mean()

    return bannerHeight, leftBannerDist, middleBannerDist, rightBannerDist


def compute_banner_model_params(
    camParamsPerImage_, maskPath_, imgWidth_, imgHeight_, nWorkers, nFrames
):
    global camParamsPerImage, maskPath, imgWidth, imgHeight
    camParamsPerImage = camParamsPerImage_
    maskPath = maskPath_
    imgWidth = imgWidth_
    imgHeight = imgHeight_

    with Pool(nWorkers) as p:
        res = list(
            tqdm(
                p.imap(compute_banner_model_params_worker, range(nFrames)),
                total=nFrames,
                desc="Computing banner model parameters",
            )
        )
    with warnings.catch_warnings():
        # If a side is not visible in the video, the nanmean over the corresponding
        # side dist array, composed of only nan values, resuls in a nan mean and
        # a RuntimeWarning. We ignore this warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bannersObjPts = np.nanmean(np.array(res), axis=0)
    bannerHeight, leftBannerDist, middleBannerDist, rightBannerDist = bannersObjPts
    # default values for when the side is not visible
    if np.isnan(leftBannerDist):
        # A bit more than half the length (105 meters) of the pitch
        leftBannerDist = -60
    if np.isnan(middleBannerDist):
        # A bit more than half the width (68 meters) of the pitch
        middleBannerDist = -40
    if np.isnan(rightBannerDist):
        # A bit more than half the length (105 meters) of the pitch
        rightBannerDist = 60
    sides = ["left", "middle", "right"]
    bannersObjPts = dict()
    for side in sides:
        if side == "middle":
            leftx = leftBannerDist
            rightx = rightBannerDist
            y = middleBannerDist
            bannersObjPts[side] = np.array(
                [
                    [leftx, y, 0],
                    [leftx, y, -bannerHeight],
                    [rightx, y, -bannerHeight],
                    [rightx, y, 0],
                ]
            )
        else:
            x = leftBannerDist if side == "left" else rightBannerDist
            y = middleBannerDist
            if side == "right":
                y = -y
            bannersObjPts[side] = np.array(
                [[x, -y, 0], [x, -y, -bannerHeight], [x, y, -bannerHeight], [x, y, 0]]
            )
    return bannersObjPts, bannerHeight


# Variant of the function above that uses the corners of the banner to compute the banner model parameters
# Works well only when 95% of the time the corners are detected and the banners in the video are perpendicular
# at the corner intersection.
def compute_banner_model_params_using_corners(
    camParamsPerImage_, maskPath_, imgWidth_, imgHeight_, nWorkers, nFrames
):
    global camParamsPerImage, maskPath, imgWidth, imgHeight
    camParamsPerImage = camParamsPerImage_
    maskPath = maskPath_
    imgWidth = imgWidth_
    imgHeight = imgHeight_

    with Pool(nWorkers) as p:
        res = list(
            tqdm(
                p.imap(
                    compute_banner_model_params_worker_using_corners, range(nFrames)
                ),
                total=nFrames,
                desc="Computing banner model parameters",
            )
        )
    with warnings.catch_warnings():
        # If a side is not visible in the video, the nanmean over the corresponding
        # side dist array, composed of only nan values, resuls in a nan mean and
        # a RuntimeWarning. We ignore this warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bannersObjPts = np.nanmean(np.array([r["bannersDims"] for r in res]), axis=0)
    bannerHeight, leftBannerDist, middleBannerDist, rightBannerDist = bannersObjPts
    # default values for when the side is not visible
    if np.isnan(leftBannerDist):
        # A bit more than half the length (105 meters) of the pitch
        leftBannerDist = -60
    if np.isnan(middleBannerDist):
        # A bit more than half the width (68 meters) of the pitch
        middleBannerDist = -40
    if np.isnan(rightBannerDist):
        # A bit more than half the length (105 meters) of the pitch
        rightBannerDist = 60
    sides = ["left", "middle", "right"]
    bannersObjPts = dict()
    for side in sides:
        if side == "middle":
            leftx = leftBannerDist
            rightx = rightBannerDist
            y = middleBannerDist
            bannersObjPts[side] = np.array(
                [
                    [leftx, y, 0],
                    [leftx, y, -bannerHeight],
                    [rightx, y, -bannerHeight],
                    [rightx, y, 0],
                ]
            )
        else:
            x = leftBannerDist if side == "left" else rightBannerDist
            y = middleBannerDist
            if side == "right":
                y = -y
            bannersObjPts[side] = np.array(
                [[x, -y, 0], [x, -y, -bannerHeight], [x, y, -bannerHeight], [x, y, 0]]
            )
    LeftCornerImgPtPerImage = np.array([r["corners"][0] for r in res])
    RightCornerImgPtPerImage = np.array([r["corners"][1] for r in res])
    return (
        bannersObjPts,
        LeftCornerImgPtPerImage,
        RightCornerImgPtPerImage,
        bannerHeight,
    )


def compute_banner_model_params_worker_using_corners(i):
    cam = Camera()
    cam.from_json_parameters(camParamsPerImage[i])
    binMask = (
        cv2.imread(maskPath + str(i).zfill(6) + ".png", cv2.IMREAD_GRAYSCALE)
        .__gt__(0)  # type: ignore
        .astype(np.uint8)
    )
    firstNonZeroIdx = np.argmax(binMask, axis=0)
    colsWithNonZero = np.where(firstNonZeroIdx != 0)[0]
    flipped = cv2.flip(binMask, 0)
    flippedLastNonZeroIdx = np.argmax(flipped, axis=0)
    # colsWithNonZero is the set of columns that have two non-zero pixels
    # and the non-zero pixels are not at the top or bottom of the image
    colsWithNonZero = np.intersect1d(
        colsWithNonZero, np.where(flippedLastNonZeroIdx != 0)[0]
    )
    firstNonZeroIdx = firstNonZeroIdx[colsWithNonZero]
    flippedLastNonZeroIdx = flippedLastNonZeroIdx[colsWithNonZero]
    lastNonZeroIdx = imgHeight - flippedLastNonZeroIdx - 1
    imgPts = np.array([[x, y] for x, y in zip(colsWithNonZero, lastNonZeroIdx)])
    objPts = np.array(
        [cam.unproject_point_on_planeZ0(p, undistort=False) for p in imgPts]
    )
    objPts[:, 2] = -1
    R, c, cameraMatrix = cam.rotation, cam.position, cam.calibration
    rvec = cv2.Rodrigues(R)[0]
    tvec = -R @ c.reshape(-1, 1)  # type: ignore
    z1m = cv2.projectPoints(objPts, rvec, tvec, cameraMatrix, np.zeros(4))[0][:, 0, 1]
    realZ = (lastNonZeroIdx - firstNonZeroIdx) / (lastNonZeroIdx - z1m)
    bannerHeight = realZ.mean()

    cornerCols = colsWithNonZero[lastNonZeroIdx == np.min(lastNonZeroIdx)]
    # check that cols is an array of consecutive integers
    # If that is not the case, it means we are in the case where two sides (for example middle and right)
    # of the banner are visible in the image but not the corner at their intersection.
    # Note that, with the "keep biggest blob" filtering, this case should not happen since, if the corner is not visible
    # but the two sides are, then there is two blobs and the biggest one is kept.
    # But in case this filtering is not applied because it is more harmful than useful, we need to handle this case.
    if np.all(np.diff(cornerCols) == 1):
        # The corner is in the image
        pixelX = cornerCols.mean().round().astype(int)
        pixelY = lastNonZeroIdx[np.where(colsWithNonZero == pixelX)[0][0]]
        cornerImgPt = np.array([pixelX, pixelY])
    else:
        # The corner is not in the image
        cornerImgPt = None

    leftCornerImgPt, rightCornerImgPt = [np.nan, np.nan], [np.nan, np.nan]
    leftBannerDist, middleBannerDist, rightBannerDist = np.nan, np.nan, np.nan

    if cornerImgPt is not None:
        cornerPitchPtX, cornerPitchPtY = cam.unproject_point_on_planeZ0(cornerImgPt)[:2]
        if (
            cornerPitchPtY < -68 / 2
            and (cornerPitchPtX < -105 / 2 or cornerPitchPtX > 105 / 2)
            and not (
                np.any([i in cornerCols for i in range(5)])
                or np.any([i in cornerCols for i in range(imgWidth - 5, imgWidth)])
            )
        ):
            # Then the corner is valid and 2 sides are visible on the image
            if cornerPitchPtX < 0:
                leftCornerImgPt = cornerImgPt
                # The left and middle sides are visible
                cond = colsWithNonZero < pixelX
                leftBannerDist = objPts[cond][:, 0].mean()
                middleBannerDist = objPts[~cond][:, 1].mean()
            else:
                rightCornerImgPt = cornerImgPt
                # The middle and right sides are visible
                cond = colsWithNonZero > pixelX
                rightBannerDist = objPts[cond][:, 0].mean()
                middleBannerDist = objPts[~cond][:, 1].mean()
            return dict(
                bannersDims=[
                    bannerHeight,
                    leftBannerDist,
                    middleBannerDist,
                    rightBannerDist,
                ],
                corners=[leftCornerImgPt, rightCornerImgPt],
            )
    # Else: when there is no corner in the image or the corner is not valid, only one side/banner is visible
    y = objPts[:, 1].mean()
    if y < -68 / 2:
        # The middle side is visible
        middleBannerDist = objPts[:, 1].mean()
    else:
        x = objPts[:, 0].mean()
        if x < 0:
            # The left side is visible
            leftBannerDist = objPts[:, 0].mean()
        else:
            # The right side is visible
            rightBannerDist = objPts[:, 0].mean()
    return dict(
        bannersDims=[bannerHeight, leftBannerDist, middleBannerDist, rightBannerDist],
        corners=[leftCornerImgPt, rightCornerImgPt],
    )


def composite_logo_into_video(
    logo_path: str,
    img_path: str,
    mask_path: str,
    cam_params_per_image: list,
    img_width: int,
    img_height: int,
    n_workers: int,
    n_frames: int,
    fps: int,
    speed_: float,
    bannersObjPts_,
    bannerHeight_,
    videoName: str,
):
    global camParamsPerImage, maskPath, imgWidth, imgHeight, nFrames
    camParamsPerImage = cam_params_per_image
    maskPath = mask_path
    imgWidth = img_width
    imgHeight = img_height
    nFrames = n_frames
    global imgPath, logo, bannersObjPts, logoWidthInMeters, speed
    imgPath = img_path
    logo = cv2.imread(logo_path)
    bannersObjPts = bannersObjPts_
    logoWidthInMeters = bannerHeight_ * logo.shape[1] / imgHeight
    speed = speed_ * logo.shape[1] / fps / logoWidthInMeters

    hsv_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_logo)
    meanAdVal = np.mean(v) * 0.96  # type: ignore
    meanAdSat = np.mean(s) * 0.80  # type: ignore
    s = meanAdSat * s / np.mean(s)  # type: ignore
    v = meanAdVal * v / np.mean(v)  # type: ignore
    s = np.clip(s, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    hsv_logo = cv2.merge([h, s, v])
    logo = cv2.cvtColor(hsv_logo, cv2.COLOR_HSV2BGR)
    # Pixels with colour [0, 0, 0] are considered the "background" and will be changed to the main logo colour afterwards.
    # To avoid changing the logo colour, we change the logo pixels whose value is [0, 0, 0] to [1, 1, 1]
    logo[np.where((logo == [0, 0, 0]).all(axis=2))] = [1, 1, 1]

    with Pool(n_workers) as p:
        _ = list(
            tqdm(
                p.imap(compositing_worker, range(nFrames)),
                total=nFrames,
                desc="Compositing logo into video",
            )
        )
    out = cv2.VideoWriter(
        f"work_dir/{videoName}", cv2.VideoWriter_fourcc(*"mp4v"), fps, (imgWidth, imgHeight)  # type: ignore
    )
    for i in tqdm(range(nFrames), desc="Saving video"):
        img = cv2.imread("work_dir/output/" + str(i).zfill(6) + ".png")
        out.write(img)
    out.release()


def compositing_worker(i):
    cam = Camera()
    sides = ["left", "middle", "right"]
    # print(i)
    logoBanners = dict()
    cut = int(round(i * speed)) % logo.shape[1]
    for side in sides:
        if side == "middle":
            logoBannerLen = (
                bannersObjPts["middle"][2, 0] - bannersObjPts["middle"][0, 0]
            )
        else:
            logoBannerLen = -2 * bannersObjPts["middle"][0, 1]
        bannerBeginning = cut
        logoBannerLen -= (
            (logo.shape[1] - bannerBeginning) / logo.shape[1]
        ) * logoWidthInMeters
        nLogosInBanner = np.floor(logoBannerLen / logoWidthInMeters).astype(np.int32)
        rest = logoBannerLen / logoWidthInMeters - nLogosInBanner
        cut = np.round(rest * logo.shape[1]).astype(np.int32)
        banner = np.concatenate(
            [logo[:, bannerBeginning:]]
            + [logo for _ in range(nLogosInBanner)]
            + [logo[:, :cut]],
            axis=1,
        )
        banner = cv2.resize(
            banner, (32766, int(32766 * (banner.shape[0] / banner.shape[1])))
        )
        logoBanners[side] = banner

    img = cv2.imread(imgPath + str(i).zfill(6) + ".png")
    cam.from_json_parameters(camParamsPerImage[i])
    mask = cv2.imread(maskPath + str(i).zfill(6) + ".png", cv2.IMREAD_GRAYSCALE)
    projectedBanners = np.zeros_like(img)
    for side in sides:
        imgPts = np.array(
            [cam.project_point(p, distort=False)[:2] for p in bannersObjPts[side]]
        )
        imgPts[1][0] = imgPts[0][0]
        imgPts[2][0] = imgPts[3][0]
        if (
            np.any(np.all(imgPts == 0, axis=1) == True)
            or (imgPts[0][0] < 0 and imgPts[3][0] < 0)
            or (imgPts[0][0] >= img.shape[1] and imgPts[3][0] >= img.shape[1])
        ):  # banner is not in the image
            continue

        tmpHeight, tmpWidth = logoBanners[side].shape[:2]
        dstPts = np.array(imgPts, dtype=np.float32)

        increase = 50  # ! Maybe try with smg like int(tmpHeight * 0.12)
        srcPts = np.array(
            [
                [0, tmpHeight - 1 + increase],
                [0, increase],
                [tmpWidth - 1, increase],
                [tmpWidth - 1, tmpHeight - 1 + increase],
            ],
            dtype=np.float32,
        )
        Minverse = cv2.getPerspectiveTransform(dstPts, srcPts)
        M = cv2.getPerspectiveTransform(srcPts, dstPts)

        warpedMask = cv2.warpPerspective(
            mask, Minverse, (tmpWidth, tmpHeight + 2 * increase)
        )
        binWarpedMask = warpedMask > 0  # type: ignore
        y, x = np.where(binWarpedMask)
        if len(y) == 0 or len(x) == 0:
            continue
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X = poly.fit_transform(x.reshape(-1, 1))
        reg = LinearRegression().fit(X, y)
        Y = reg.predict(X)
        width = np.mean(abs(y - Y)) * 4

        x = np.arange(32766, step=762)  # type: ignore
        xRegression = np.arange(32766)
        xPoly = poly.fit_transform(xRegression.reshape(-1, 1))
        yc = reg.predict(xPoly)
        yt = yc - width / 2
        yb = yc + width / 2
        topLinePts = np.float32([[x_, y_] for x_, y_ in zip(xRegression, yt)]).reshape(-1, 1, 2)  # type: ignore
        bottomLinePts = np.float32([[x_, y_] for x_, y_ in zip(xRegression, yb)]).reshape(-1, 1, 2)  # type: ignore
        topLinePts = cv2.perspectiveTransform(topLinePts, M).reshape(-1, 2)
        bottomLinePts = cv2.perspectiveTransform(bottomLinePts, M).reshape(-1, 2)
        srcPts2 = np.float32([[0, 0], [761, 0], [761, logoBanners[side].shape[0] - 1], [0, logoBanners[side].shape[0] - 1]])  # type: ignore

        for x_ in x:
            billPts = np.float32([topLinePts[x_], topLinePts[x_ + 761], bottomLinePts[x_ + 761], bottomLinePts[x_]])  # type: ignore
            M2 = cv2.getPerspectiveTransform(srcPts2, billPts)  # type: ignore
            warpedLogo = cv2.warpPerspective(
                logoBanners[side][:, x_ : x_ + 762],
                M2,
                (img.shape[1], img.shape[0]),
                flags=cv2.INTER_NEAREST,
            )  # flags=cv2.INTER_LINEAR might be better ?
            projectedBanners = cv2.bitwise_or(projectedBanners, warpedLogo)

    binMask = np.bitwise_and(mask > 0, mask < 3)  # type: ignore
    projectedBanners[np.where((projectedBanners == [0, 0, 0]).all(axis=2))] = logo[0, 0]
    img[binMask] = projectedBanners[binMask]
    img2 = deepcopy(img)
    img2[binMask] = cv2.GaussianBlur(img, (3, 3), 0)[binMask]
    alpha = 0.3
    img3 = cv2.addWeighted(img, alpha, img2, 1.0 - alpha, 0)
    # cam.draw_pitch(img3)
    # cam.draw_corners(img3)

    cv2.imwrite("work_dir/output/" + str(i).zfill(6) + ".png", img3)
