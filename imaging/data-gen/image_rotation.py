import cv2
import numpy as np
img = cv2.imread("shapes/rectangle.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv.imshow("before transform",img)
width, height = img.shape[:2]
print(width,height)
theta = -np.pi/4

# image_corners = np.array([
#     [0,0],[width,0],[width,height],[0,height]
# ], dtype=np.float32)

# transform_matrix = cv.getPerspectiveTransform(
#     src=image_corners,
#     dst=(image_corners+np.random.random(size=(4,2))*50).astype(np.float32)
# )

# print(transform_matrix)

def get_rotated_image(img: cv2.Mat, theta: float):
    '''
    Theta is the rotation direction, clockwise, in radians
    '''
    height, width = img.shape[:2]
    translate_to_center_matrix = np.array([
        [1,0,-width/2],
        [0,1,-height/2],
        [0,0,1]
    ])

    rotation_matrix = np.array([
            [np.cos(theta),-np.sin(theta),0],
            [np.sin(theta),np.cos(theta),0],
            [0,0,1]
        ],
        dtype=np.float32
    )

    new_w = np.ceil(width*np.abs(np.cos(theta))+height*np.abs(np.sin(theta))).astype(int)
    new_h = np.ceil(height*np.abs(np.cos(theta))+width*np.abs(np.sin(theta))).astype(int)

    undo_translate_to_center_matrix = np.array([
        [1,0,new_w/2],
        [0,1,new_h/2],
        [0,0,1]
    ])

    return cv2.warpPerspective(
    src=img, 
    M=undo_translate_to_center_matrix@rotation_matrix@translate_to_center_matrix,
    dsize=(new_w,new_h)
)

def get_shape_bbox(img: cv2.Mat, show_box=False):
    '''
    img: cv2 Mat with shape (width,height). Can't have a third channel, has to be grayscale

    Gets the bounding box for a shape on a completely black background (all background values are 0 exactly)
    Runtime is proportional to the number of pixels in the image (width*height).
    Returns a tuple (x_min, y_min, x_max, y_max)

    If show_box is True, this will use cv.imshow to display the bounding box on the image. It will use cv.waitKey(0) to wait for a keypress before dismissing the image display window.

    '''
    width,height = img.shape[:2]

    x_min, y_min = width, height
    x_max, y_max = 0,0

    for row in range(height):
        found_shape=False
        for col in range(width):
            if img[col][row]>0:
                x_min = min(col, x_min)
                x_max = max(col, x_max)
                found_shape=True
                break
        if found_shape:
            for col in reversed(range(width)):
                if img[col][row]>0:
                    x_min = min(col, x_min)
                    x_max = max(col, x_max)
                    break
            y_max = max(row, y_max)
            y_min = min(row, y_min)
            
    if show_box:
        with_rectangle = img.copy()
        cv2.rectangle(with_rectangle, (y_min, x_min),(y_max, x_max), (255,0,0))
        
        cv2.imshow("debug view of rotation",with_rectangle)
        cv2.waitKey(0)
    
    return (x_min, y_min, x_max, y_max)

if __name__=="__main__":
    warped = get_rotated_image(img,theta)
    box = get_shape_bbox(warped)
    print(box)