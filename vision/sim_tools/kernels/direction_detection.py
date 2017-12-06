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

def direction_subsamp(width, dir_ang, weight, delay=1):
    if dir_ang%180 == 90 and dir_ang != 0: # horizontal motion
        krn = np.ones((width, 1))*weight
        dly = np.ones((width, 1))*delay
    elif (dir_ang)%180 == 0: # vertical motion
        krn = np.ones((1, width))*weight
        dly = np.ones((1, width))*delay

    print(krn)
    print(dly)

    return krn, dly

def direction_kernel(width, height, min_delay=1, weight=2.,
                     dir_ang=0., delta_ang=15.,
                     delay_func=lambda x: x, 
                     weight_func=lambda d, a, w: w/(a*d),
                     use_euclidean_dist=False,
                     max_delay=144.):
    min_ang = dir_ang - delta_ang
    max_ang = dir_ang + delta_ang
    hw = width//2
    hh = height//2
    weights = np.zeros((height, width))
    delays  = np.zeros((height, width))
    dists = np.zeros((height, width))
    angs = np.zeros((height, width))
    for r in range(height):
        dr = hh - r
        for c in range(width):
            dc = hw - c
            if use_euclidean_dist:
                dist = np.sqrt(dr**2 + dc**2)
            else:
                dist = np.abs(dr) + np.abs(dc)

            if dc == 0 and dr == 0:
                ang = dir_ang
            else:
                # ang = np.rad2deg( np.arctan2(np.abs(dc), np.abs(dr)) )
                if np.abs(dir_ang) == 90.0:
                    ang = np.rad2deg( np.arctan2(dc, dr) )
                else:
                    ang = np.rad2deg(np.arctan2(np.abs(dc), dr))

            if not (min_ang < ang < max_ang):
                continue
            # if not (0 < ang < delta_ang):
            #     continue

            ang = np.abs(ang - dir_ang)

            delay = np.round(min_delay + delay_func(dist, ang))

            if delay > max_delay:
                continue

            w = weight_func(dist, ang, weight)
            # print(dist, weight, ang)
            weights[r, c] = w
            # print(weights[r, c])

            delays[r, c]  = delay

            angs[r, c] = ang
            dists[r, c] = dist

    # print(min_ang, dir_ang, max_ang)
    print(weights)
    print(delays)
    # print(np.round(15./angs))
    # print(np.round(1+angs/45.))
    # print(dists)
    # print(np.round(dists*1.5 + np.round(15./angs)))
    # import sys
    # sys.exit(0)
    return weights, delays