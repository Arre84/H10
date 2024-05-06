import numpy as np
import cv2 as cv
import argparse
from numpy.typing import NDArray

previous = []
left = 0
right = 0

def user_interaction() ->argparse.ArgumentParser:
    """
    Interact with the user to get the path of the image containing the object to track, video to do the tracking
    and resize of the frame.
    
    Returns:
        args: Path for the image and video.
    """
    parser = argparse.ArgumentParser(description='Corner Detection')
    parser.add_argument('-i', '--img_obj',
                        type=str,
                        required=True,
                        help="Path to the image file where corners will be detected")
    parser.add_argument('-o', '--video',
                        type=str,
                        required=True,
                        help="Path to the image file where corners will be detected")
    parser.add_argument('--resize',
                        type= int,
                        required= True,
                        help= "Percentage to resize the image")
    args = parser.parse_args()
    return args

def initialise_camera(args:argparse.ArgumentParser)->cv.VideoCapture:
    """
    Opens the video path or camera index provided by the user.

    args:
        args: cotains the camera index to open.
    
    Returns:
        Cap: Variable with the frame.
    """
    cap = cv.VideoCapture(args.video)
    return cap

def load_image(path:str)->cv:
    """
    Loads an image from the specified path in grayscale.
    
    Parameters:
        path: The path to the image file.
    
    Returns:
        img: The loaded image in grayscale.
    """
    img = cv.imread(path,cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("One or both images not found. Please check the paths.")
    img = resize(img,40)
    return img

def grayscale(img:NDArray) ->NDArray:
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return gray

def resize(img:NDArray, per:int) -> NDArray:
    """
    Resize an image to a specified percentage    
    Parameters:
        img: Image to perfor the resize.
        per: Percentage to which the resize will be performed
    Returns:
        img_resize: Resized image
    """
    # Resize images
    width = int(img.shape[1] * per / 100)
    height = int(img.shape[0] * per / 100)
    dim = (width, height)
    img_resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return img_resized

def extract_corners_and_descriptors(image:NDArray)-> tuple[list[cv.KeyPoint], NDArray]:
    """
    Extracts corner points and descriptors from an image using Harris corner detection.
    Parameters:
        image: Image to perform corener detection

    Returns:
        kp: List containing the points where a corener was detected.
        des: Array containing the descriptors of the provided image.
    """

    # Harris corner detection/
    harris_corners1 = cv.cornerHarris(image, blockSize=15, ksize=11, k=0.04)

# Threshold for corner detection
    threshold = 0.1 * harris_corners1.max()
    thresholded_img1 = np.zeros_like(image)
    thresholded_img1[harris_corners1 > threshold] = 255
#Extract keypoints
    keypoints1 = np.argwhere(thresholded_img1 == 255).tolist()

# Convert keypoints to list of KeyPoint objects
    keypoints1 = [cv.KeyPoint(x[1], x[0], 3) for x in keypoints1]

    # Initiate SIFT detector
    sift = cv.SIFT_create()

# compute the descriptors with BRIEF
    kp, des = sift.compute(image,keypoints1)
    return kp, des

def draw_matches(img2: np.ndarray, descriptors1: np.ndarray, descriptors2: np.ndarray, kp1: list[cv.KeyPoint], kp2: list[cv.KeyPoint]):
    """
    Draws matches between two images based on their descriptors and keypoints using RANSAC-based feature matching.

    Parameters:
        img2: Tracking frame
        descriptors1: List of descriptors of the first image
        descriptors1: List of descriptors of the second image
        kp1: List containing the keypoints of the first image
        kp2: List containing the keypoints of the second image

    Returns:
        None
    """
    # Create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Apply RANSAC to filter out outliers
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 10)

    # Filter matches using RANSAC mask
    matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]
    x_coordinates = []
    y_coordinates= []
    # Draw match points only on the second image
    img_matches = img2.copy()  # Create a copy of the second image
    for match in matches:
        pt2 = kp2[match.trainIdx].pt
        x_coordinates.append(pt2[0])  # Agregar la coordenada x a la lista
        y_coordinates.append(pt2[1])  # Agregar la coordenada y a la lista
        cv.circle(img_matches, (int(pt2[0]), int(pt2[1])),2, (0, 255, 0), -1)  # Draw a circle at the matched keypoint

    # Drawing middle line
    # Get image dimensions
    height, width, _ = img2.shape
    # Define the starting point at the top of the image (horizontally centered)
    pt1 = (int(width / 2), 0)
    # Define the ending point at the bottom of the image (horizontally centered)
    pt2 = (int(width / 2), height - 1)
    cv.line(img_matches,pt1,pt2,(255,0,0),1)

    #Draw bounding box, centroid and obtain max coordinates
    if x_coordinates and y_coordinates:
        rectangle(x_coordinates, y_coordinates, img_matches)
    #Show current frame
    cv.imshow('Matched Features', img_matches)
    return match

def rectangle(x: np.ndarray, y: np.ndarray, image: np.ndarray):
    """
    Draws the bounding box and computes and draws the centroid of the object.

    Parameters:
        x: x coordinates of the detected keypoints
        y: y coordinates of the detected keypoints
        image: frame to compute the drawings

    Returns:
        center of the object
    """
    global previous
    height, width, _ = image.shape
    xmin = int(min(x))
    xmax = int(max(x))
    ymin = int(min(y))
    ymax = int(max(y))
    pt1 = (xmin, ymin)  # Bottom-left corner
    pt2 = (xmax, ymax)  # Top-right corner
    cv.rectangle(image, pt1, pt2, (255, 0, 0), 1)
    # Calculate the center for drawing the circle
    center_x = (pt1[0] + pt2[0]) // 2
    center_y = (pt1[1] + pt2[1]) // 2
    centroid = (center_x,center_y)
    current = estimate_centroid(centroid,previous)
    if (previous):
        compute_change(image,width,current,previous)
    previous = current
    cv.circle(image, (current[0], current[1]), 2, (0, 0, 255), 2)
    return None

def estimate_centroid(current: tuple[int, int], previous: tuple[int, int], smoothing_factor: float = 0.25) -> tuple[int, int]:
    """
    Computes the centorid of the object smoothening the change between the current centroid and the previous one.

    Parameters:
        Current: Current coordinates of the centroid 
        Previous: Previous coordinates of the centroid 
        smoothing_factor: Percentage of change in centroids.
    Retruns: 
        Current: Coordinates of the new centroid.
    """
    if current and previous:
        # Interpolate between current and previous centroid positions based on the smoothing factor
        smoothed_x = int(previous[0] + (current[0] - previous[0]) * smoothing_factor)
        smoothed_y = int(previous[1] + (current[1] - previous[1]) * smoothing_factor)

        return smoothed_x, smoothed_y
    else:
        return current

def compute_change(img:NDArray, width:int, centroid:tuple[int, int],previous:tuple[int, int])->None:
    global right
    global left
    if(previous[0]<=width//2):
        if(centroid[0]>width//2):
            right += 1
    elif(previous[0]>=width//2):
        if(centroid[0]<width//2):
            left += 1
    right_show = f"Left2Right:{right}"
    left_show = f"Left2Right:{left}"
    print("P",previous)
    print("C",centroid)
    print(width//2)
    cv.putText(img,right_show,(width//40,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    cv.putText(img,left_show,(width//40,60),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    return None

def pipeline()->None:
    args = user_interaction()
    img_obj = load_image(args.img_obj)
    kp_obj, des_obj = extract_corners_and_descriptors(img_obj)
    cap = initialise_camera(args)
    while cap.isOpened():
        # Read current frame
        ret, frame = cap.read()
        # Check if the image was correctly captured
        if not ret:
            print("It seems like a problem has occured, try running the program again, in case the\n"
                   "problem keeps ocurring, call : 614-345-3164")
            break
        rescale_frame = resize(frame,args.resize)
        gray_frame = grayscale(rescale_frame)
        kp, des = extract_corners_and_descriptors(gray_frame)
        draw_matches(rescale_frame, des_obj, des, kp_obj,kp)
        key = cv.waitKey(20)
        if key == ord('q') or key == 27:
            print("Programm finished!")
            break
    cv.destroyAllWindows()
    cap.release()
    return None

if __name__ == "__main__":
    pipeline()