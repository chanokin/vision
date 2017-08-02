# import vision.vision.sim_tools.connectors.direction_connectors as dir_conns
import numpy as np

def dir_to_ang(direction):
    ang = 0
    if   direction.lower() == 'w'  or direction.lower() == 'west':
        ang = 0
    elif direction.lower() == 'sw' or direction.lower() == 'south west':
        ang = 45
    elif direction.lower() == 's'  or direction.lower() == 'south':
        ang = 90
    elif direction.lower() == 'se' or direction.lower() == 'south east':
        ang = 135
    elif direction.lower() == 'e'  or direction.lower() == 'east':
        ang = 180
    elif direction.lower() == 'ne' or direction.lower() == 'north east':
        ang = 225
    elif direction.lower() == 'n'  or direction.lower() == 'north':
        ang = 270
    elif direction.lower() == 'nw' or direction.lower() == 'north west':
        ang = 315
    else:
        np.random.seed()
        ang = np.random.randint(0, 360)
        print("Random angle %s"%ang)
    
    ang -= 90
    return ang

def direction_kernel(width, height, min_delay=1, weight=2.,
                     dir_ang=0., delta_ang=15.,
                     delay_func=lambda x: x, 
                     weight_func=lambda d, a, w: w/(a*d)):
    min_ang = dir_ang - delta_ang
    max_ang = dir_ang + delta_ang
    hw = width//2
    hh = height//2
    weights = np.zeros((height, width))
    delays  = np.ones((height, width))

    for r in range(height):
        dr = hh - r
        for c in range(width):
            dc = hw - c
            dist = np.sqrt(dr**2 + dc**2)
            ang = np.rad2deg( np.arctan2(dc, dr) )
            if ang > max_ang or min_ang > ang:
                continue

            weights[r, c] = weight_func(dist, ang, weight)
            delays[r, c]  = np.round(min_delay + delay_func(dist))

    return weights, delays