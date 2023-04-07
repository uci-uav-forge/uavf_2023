import os
import numpy as np
import cv2
from text_rendering import get_shape_text_area_from_mat

def create_obj_file(shape_img: cv2.Mat, filename: str):
    """
    Create an obj file of the polygon defined by the given coordinates.
    `coordinates` is a ndarray of shape (n, 2) where n is the number of vertices.
    """
    coordinates = get_polygon(shape_img)
    shape_text_bbox = get_shape_text_area_from_mat(shape_img)
    text_width = shape_text_bbox[1][0] - shape_text_bbox[0][0]
    text_height = shape_text_bbox[1][1] - shape_text_bbox[0][1]
    text_center_x = (shape_text_bbox[1][0] + shape_text_bbox[0][0])/2
    text_center_y = (shape_text_bbox[1][1] + shape_text_bbox[0][1])/2
    shape_center_x = shape_img.shape[1]/2
    shape_center_y = shape_img.shape[0]/2
    for point in coordinates:
        point[0] -= shape_img.shape[1]/2 + (text_center_x - shape_center_x)
        point[1] -= shape_img.shape[0]/2 + (text_center_y - shape_center_y)
    normalized_coords = (coordinates)/max(text_width,text_height)
    with open(filename, 'w') as f:
        for point in normalized_coords:
            f.write(f'v {point[0]} 0 {point[1]}\n')
        coord_indices_list = list(range(len(normalized_coords)))
        triangles = []
        for _ in range(len(normalized_coords)-2):
            for i in range(len(coord_indices_list)):
                ear_tip = coord_indices_list[i]
                prev = coord_indices_list[(i-1)%len(coord_indices_list)]
                next = coord_indices_list[(i+1)%len(coord_indices_list)]
                p1, p2, p3 = normalized_coords[prev], normalized_coords[ear_tip], normalized_coords[next]
                cross = np.cross(p2-p1, p3-p2) 
                if cross > 0: continue
                
                disqualified = False
                for j in coord_indices_list:
                    if j in [prev, ear_tip, next]:
                        continue
                    if cv2.pointPolygonTest(np.array([p1,p2,p3], dtype=np.float32), normalized_coords[j], False) >= 0:
                        disqualified = True
                        break
                if not disqualified:
                    triangles.append([prev,ear_tip,next])
                    coord_indices_list.remove(ear_tip)
                    break
        for triangle in triangles:
            f.write(f'f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n')            

def get_polygon(shape_img: cv2.Mat) -> np.ndarray:
    '''
    Returns the enclosing polygon of the shape in the image. The polygon is a list of points, each point being a list of 2 coordinates. (ndarray shape (n,2))
    '''
    im = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)
    im = cv2.threshold(im, 1, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    # imshow the contours on a black iamge
    # prev = np.zeros(shape_img.shape, dtype=np.uint8)
    # cv2.drawContours(prev, contours[0], -1, (255,255,255), 3)
    # cv2.imshow('contours', prev)
    # cv2.waitKey(0)
    return np.array(contours[0]).reshape(-1,2)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(f'{base_dir}/shapes/3d', exist_ok=True)
    shape_filenames = os.listdir(f'{base_dir}/shapes')
    for shape in shape_filenames:
        if not shape.endswith('.png'):
            continue
        shape_img = cv2.imread(f'{base_dir}/shapes/{shape}')
        create_obj_file(shape_img, f'{base_dir}/shapes/3d/{shape[:-4]}.obj')