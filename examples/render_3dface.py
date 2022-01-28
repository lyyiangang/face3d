import numpy as np
import cv2
import scipy.io as sio
from skimage import io
import sys
sys.path.append('..')
import face3d
from face3d import mesh

def display(raw_rgb_img):
    img = (raw_rgb_img * 255).astype(np.uint8)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('bgr_img', bgr_img)
    cv2.waitKey(0)

def run():
    """
    world coordinate system:
         y
         |
         |
         |
         |
         |
         O-------------------x
        /
       /
      /
     /
    /
   z
    -------------
    """
    h, w = 256, 256
    ori_data = sio.loadmat('Data/example1.mat')
    vertices = ori_data['vertices']
    colors, triangles = ori_data['colors'], ori_data['triangles']
    colors = colors/np.max(colors)# map to [0, 1]
    vertices -= np.mean(vertices, 0)[None, :] # move face coordinate system center to world cs center

    # convert obj to world cs 
    obj = {}
    # scale face model to real size
    obj['s'] = h /(np.max(vertices[:,1]) - np.min(vertices[:,1])) 
    obj['angles'] = [0, 0, 0]
    obj['t'] = [0, 0, 0]
    # convert face to world coordinate system
    R = mesh.transform.angle2matrix(obj['angles'])
    # s*X*R + t
    world_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])

    # define camera in world cs
    camera = {}
    camera['proj_type'] = 'orthographic' # perspective or orthographic
    camera['eye'] = [0, 0, 500] # eye location in world coordinate system
    camera['at'] = [0, 0, 0] # gaze at position
    camera['up'] = [0, 1, 0]
    # for perspective case
    camera['near'] = 1000
    camera['far'] = -100
    camera['fovy'] = 30

    # add 1 point light
    light_intensities = np.array([[1, 1, 1]], dtype = np.float32)
    light_positions = np.array([[0, 0, 300]])
    # add 2 point lights
    # light_intensities = np.array([[1, 1, 1], [1, 1, 1]], dtype = np.float32)
    # light_positions = np.array([[300, 0, 300], [-300, 0, 300]])
    colors = mesh.light.add_light(world_vertices, triangles, colors, light_positions, light_intensities)

    if camera['proj_type'] == 'orthographic':
        camera_vertices = mesh.transform.orthographic_project(world_vertices) # not normalized
        projected_vertices = mesh.transform.lookat_camera(camera_vertices, camera['eye'], camera['at'], camera['up'])
    else:
        # to camera coordinate system
        camera_vertices = mesh.transform.lookat_camera(world_vertices, camera['eye'], camera['at'], camera['up'])
        # perspective project and convert to NDC. positon is normlized to [-1, 1]
        projected_vertices = mesh.transform.perspective_project(camera_vertices, camera['fovy'], near = camera['near'], far = camera['far'])
    # to image coords(position in image), if perspectvie, the coordinate will be map to image's w/h
    image_vertices = mesh.transform.to_image(projected_vertices, h, w, camera['proj_type'] == 'perspective')

    rendering = mesh.render.render_colors(image_vertices, triangles, colors, h, w)
    raw_rgb_img = np.minimum((np.maximum(rendering, 0)), 1)
    display(raw_rgb_img)

if __name__ == '__main__':
    run()