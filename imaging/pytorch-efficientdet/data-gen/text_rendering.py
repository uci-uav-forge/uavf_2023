import cv2
import numpy as np
import bisect
from functools import cache
from itertools import product

@cache
def _get_text_scale(letter: str, max_w: int, max_h:int, thickness: int):
    def bisection_key(scale):
        size, _ = cv2.getTextSize(letter,0,scale,thickness)
        w,h=size
        if w<max_w  and h< max_h:
            return 0
        else:
            return 1
    search_space_size=100
    scales = np.linspace(0,10,search_space_size)
    scale_idx = bisect.bisect(list(map(bisection_key,scales)),0,0,search_space_size)
    optimal_scale = scales[scale_idx-1]
    size, _ = cv2.getTextSize(letter,0,optimal_scale,thickness)
    return optimal_scale, size

def drawText(img: cv2.Mat, letter: str, bounding_box: "tuple[tuple[int]]", thickness: int, color=(0,0,0), fontFace=cv2.FONT_HERSHEY_SIMPLEX):
    '''
        `bounding_box` should be [top_left, bot_right] and each point should be [x,y]


        WARNING:
        This produces text that's at most 250 pixels in its largest axis because the max scale it uses for `cv2.putText` is 10 and the default font characters are about 25 pixels large at scale 1.
    '''

    '''
        Internal notes:
        opencv default text size with scale 1 is around 25x25 at most for a letter
        Z is 15 wide, m is 24 wide, A is 22 tall.
    '''
    t_left, b_right = bounding_box
    optimal_scale, size = _get_text_scale(letter, b_right[0]-t_left[0], b_right[1]-t_left[1], thickness)
    margin_x, margin_y = [b_right[i]-t_left[i]-size[i] for i in (0,1)]
    cv2.putText(img, 
                text=letter, 
                org=(t_left[0]+margin_x//2,b_right[1]-margin_y//2),
                fontFace=fontFace,
                fontScale=optimal_scale,
                color=color,
                thickness=thickness)

@cache
def _get_shape_text_area(shape_name: str):
    img = cv2.imread(f"shapes/{shape_name}.png")
    h,w, _c = img.shape
    bounding_box = []
    for i,j in product(range(h),range(w)):
        if img[i][j][0]>0:
            bounding_box.append((j,i))
            break
    for i,j in product(reversed(range(h)),reversed(range(w))):
        if img[i][j][0]>0:
            bounding_box.append((j,i))
            break
    return bounding_box

def get_shape_text_area(shape_name: str):
    # implementation is in the private function so that the @cache doesn't mess up the argument type hint
    return _get_shape_text_area(shape_name)
    

if __name__=="__main__":
    # print(get_shape_text_area("circle"))
    img = np.ones((1000,1000)) * 255
    img = cv2.resize(img, (200,100))
    print(img.shape)
    # letter="A"
    # bbox = [(100,100),(130,130)]
    # drawText(img, letter, bbox, 3)
    # cv2.rectangle(img, bbox[0],bbox[1],(0,0,0),2)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)