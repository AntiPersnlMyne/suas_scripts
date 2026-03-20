import os
import numpy as np
import cv2 as cv
import glob
from numpy.typing import NDArray

import traceback

from flir import RJPEG

def read_img(path, coeff:NDArray):
    """Read RJPEG image and apply radiometric coefficients"""
    return RJPEG(path).raw_counts * coeff[:,:,0] + coeff[:,:,1]

def register_video(datadir, outpath, metadatapath, calcoeffpath, altitude_change_frames, angle_change_frames, ref_every):

    # Load radiometric calibration coefficients
    calibration_coeffs = np.load(calcoeffpath)

    # Load location data
    metadata = np.load(metadatapath)

    # This is what's in the metadata file
    #altitude_abs = metadata[:,0]
    #altitude_rel = metadata[:,1]
    #gimbal_roll = metadata[:,2]
    #gimbal_yaw = metadata[:,3]
    #gimbal_pitch = metadata[:,4]
    #flight_roll = metadata[:,5]
    #flight_yaw = metadata[:,6]
    #flight_pitch = metadata[:,7]

    # Find video frames
    image_files = sorted(glob.glob(os.path.join(datadir, "*.jpg")))

    # Altitude change segment of video 
    frames_alt = [image_files[i] for i in altitude_change_frames]
    metadata_alt = metadata[altitude_change_frames]
    sort_idxs = (-metadata_alt[:,1]).argsort() # Sort by altitude descending
    metadata_alt = metadata[sort_idxs]
    frames_alt = [frames_alt[i] for i in sort_idxs]

    # View angle change segment of video
    frames_ang = [image_files[i] for i in angle_change_frames]
    metadata_ang = metadata[angle_change_frames]
    sort_idxs = metadata_ang[:,4].argsort() # Sort by view angle
    metadata_ang = metadata[sort_idxs]
    frames_ang = [frames_ang[i] for i in sort_idxs]

    # Put together all the sorted data
    metadata_total = np.concat([metadata_ang, metadata_alt], axis=0)
    frames = frames_ang + frames_alt

    # At least MIN_MATCH_COUNT features must be matched to compute transformation matrix
    MIN_MATCH_COUNT = 10

    # Initialize SIFT
    sift = cv.SIFT_create()

    # Initialize FLANN for feature matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params,search_params)

    ref_img_i = len(frames_ang)
    ref_img = read_img(frames[ref_img_i], calibration_coeffs)

    # Function to convert images to 8 bit
    norm_range = np.quantile(ref_img, [0.001, 0.999]) # Exclude very extreme outlier pixels, close to min max normalization though
    def to8bit(img):
        img_normalized = np.clip(img, *norm_range)
        img_normalized = (img_normalized - np.min(img_normalized)) / (np.max(img_normalized) - np.min(img_normalized))
        img_uint8 = (255*img_normalized).astype(np.uint8)
        return img_uint8

    ref_img = to8bit(ref_img)
    ref_kp, ref_des = sift.detectAndCompute(ref_img, None)

    transform_matrices = np.zeros((len(image_files), 3, 3), dtype=np.float64)
    #registered_to = np.zeros((len(image_files),), dtype=np.uint64)
    registered_to = np.full((len(image_files),), ref_img_i, dtype=np.uint64)

    print("Registering multiple view angles.")
    for i, image_file in enumerate(frames):
        if i == ref_img_i:
            print("Regsitering multiple altitudes.")
            continue # Skip first reference image

        img = to8bit(read_img(image_file, calibration_coeffs))
        kp, des = sift.detectAndCompute(img, None)

        try:
            # Match features
            matches = flann.knnMatch(ref_des, des, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for nn in matches: # Iterate through the nearest neighbor matches
                if len(nn) == 2: # Verify FLANN found 2 nearest neighbors
                    if nn[0].distance < 0.7*nn[1].distance:
                        good.append(nn[0])
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([ ref_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                # Find transformation matrix from reference to current frame
                M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
                transform_matrices[i] = M
            else:
                print("Error registering frame", image_file, "to frame", image_files[ref_img_i])
                print(f"Not enough matches found. # Matches = {len(good)}, # KP = {len(kp)}, # Ref KP = {len(ref_kp)}")
                continue
        except Exception as e:
            traceback.print_exc()
            print("Error registering frame", image_file, "to frame", image_files[ref_img_i])
            for x in matches:
                if len(x) != 2:
                    print(x)
            continue

        registered_to[i] = ref_img_i
        
        print(f"{i}/{len(image_files)}, matches={len(good)}")

        # Update reference frame if in altitude change timeframe
        j = i - len(frames_ang)
        if (j >= 0) and (not j % ref_every):
            ref_img_i = i
            ref_img = img
            ref_kp = kp
            ref_des = des
            print(f"New reference frame: {i}")

    np.savez(outpath, transform_matrices=transform_matrices, registered_to=registered_to, files=np.array(frames), metadata=metadata_total)
    print("Saved registration transform matrices to", outpath)

if __name__ == "__main__":
    datadir = "/home/star/data/high_gain/flight3/"
    metadatapath = "./high_gain_flight3_save/flight_metadata.npy"
    calcoeffpath = "./calibration/20251206_1200_10ec_10_70x5_manual_coefficients.npy"
    outpath = "./high_gain_flight3_save/flight_registration_combined.npz"

    altitude_change_frames = range(45, 86)
    angle_change_frames = range(0, 45)

    ref_every = 5
    
    register_video(datadir, outpath, metadatapath, calcoeffpath, altitude_change_frames, angle_change_frames, ref_every)
