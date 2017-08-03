from ..common import *
from mapping_funcs import row_col_to_input

def competition_connector(img_w, img_h,
                        pre_shape, pre_starts, pre_steps,
                        post_shape, post_starts, post_steps, kernel, 
                        exc_delay=3., inh_delay=1.):
    # pre == source == s
    # post == target == t
    # 0 == HEIGHT == ROWS
    # 1 == WIDTH  == COLUMNS

    hh = kernel.shape[0]//2
    hw = kernel.shape[1]//2

    pre_to_post = {}
    for tr in range(post_shape[0]):
        otr = tr + 0
        for tc in range(post_shape[1]):
            otc = tc + 0
            post = tr*post_shape[1] + tc
            
            # tr and tc are currently in sub-sampled space
            # we need to convert to input space
            tr = tr * post_steps[0] + post_starts[0]
            if tr < 0 or tr >= img_h:
                continue
            
            tc = tc * post_steps[1] + post_starts[1]
            if tc < 0 or tc >= img_w:
                continue

            for kr in range(-hh, hh):
                sr = tr + kr

                if sr < 0 or sr >= img_h:
                    continue

                for kc in range(-hw, hw):
                    sc = tc + kc

                    if sc < 0 or sc >= img_w:
                        continue

                    # print("from (%d, %d) => (%d, %d)"%
                    #       (sr, sc, tr, tc))
                    
                    # sr and sc are in input space, 
                    # we need to convert to sub-sampled space
                    sr = subsamp_size(pre_starts[0], sr, pre_steps[0])
                    sc = subsamp_size(pre_starts[1], sc, pre_steps[1])

                    # print("from (%d, %d) => (%d, %d)    MAX (%d, %d) "%
                    #     (sr, sc, otr, otc, pre_shape[0], pre_shape[1]))

                    if  sr < 0 or sr >= pre_shape[0] or \
                        sc < 0 or sc >= pre_shape[1]:
                        # print("Not valid")
                        continue

                    pre = sr*pre_shape[1] + sc
                    # print("from %d => %d"%(pre, post))
                    if not ( pre in pre_to_post.keys() ):
                        pre_to_post[pre] = { post: kernel[kr+hh, kc+hw] }

                    elif not ( post in pre_to_post[pre].keys() ):
                        pre_to_post[pre][post] = kernel[kr+hh, kc+hw]

                    else:
                        pre_to_post[pre][post] += kernel[kr+hh, kc+hw]

    exc_conns = []
    inh_conns = []
    max_post = 0
    for pre in pre_to_post:
        
        if len(pre_to_post[pre].keys()) and \
           max_post < np.max(pre_to_post[pre].keys()):

            max_post = np.max(pre_to_post[pre].keys())
        # print(pre_to_post[pre])
        for post in pre_to_post[pre]:

            w = pre_to_post[pre][post]
            d = exc_delay if w > 0 else inh_delay

            if w > 0:
                # print("EXC")
                exc_conns.append((pre, post, w, d)) 
            else:
                # print("INH")
                inh_conns.append((pre, post, w, d)) 

            # print((pre, post, w, d))

    # print('max pre %d'%np.max(pre_to_post.keys()))
    # print('max post %d'%max_post)
    # print("len exc: %d - inh: %d"%(len(exc_conns), len(inh_conns)))
    return exc_conns, inh_conns


    
def full_kernel_connector(pre_layer_width, pre_layer_height, kernel, 
                exc_delay=3., inh_delay=1., col_step=1, row_step=1, 
                col_start=0, row_start=0, map_to_src=row_col_to_input,
                row_bits=8, pop_width=None, on_path=True,
                min_w = 0.0001, remove_inh_only=True):
    '''Create connection list based on a convolution kernel, the format
       for the lists is to be used with PyNN 0.7. 
       (Pre neuron index, Post neuron index, weight, delay)
       
       :param layer_width: Pre layer width
       :param layer_height: Pre layer height
       :param kernel: Convolution kernel
       :param col_step: Skip this many columns (will reduce Post layer size)
       :param row_step: Skip this many rows (will reduce Post layer size)
       :param delay: How many time units will it take the spikes to
                     get from Pre to Post

       :return exc_conns: Excitatory connections list
       :return inh_conns: Inhibitory connections list
    '''
    row_bits = row_bits if pop_width is None else pop_width
    
    layer_width, layer_height = pre_layer_width, pre_layer_height
    exc_conns = []
    inh_conns = []
    kh, kw = kernel.shape
    half_kh, half_kw = kh//2, kw//2
    
    dst_width  = subsamp_size(col_start, layer_width, col_step)
    dst_height = subsamp_size(row_start, layer_height, row_step)
    num_dst = dst_width*dst_height
    num_src = pre_layer_height*pre_layer_width
    exc_counts = [0 for dr in range(num_dst)]
    inh_counts = [0 for dr in range(num_dst)]
    
    for dr in range(row_start, layer_height, row_step):
        for dc in range(col_start, layer_width, col_step):

            sr0 = dr - half_kh
            sc0 = dc - half_kw
            drr = subsamp_size(row_start, dr, row_step)
            dcc = subsamp_size(col_start, dc, col_step)
            if drr < 0 or drr >= dst_height or \
               dcc < 0 or dcc >= dst_width:
               continue

            dst = drr*dst_width + dcc

            for kr in range(kh):
                sr = sr0 + kr
                if sr < 0 or sr >= pre_layer_height:
                    continue

                for kc in range(kw):
                    sc = sc0 + kc
                    if sc < 0 or sc >= pre_layer_width:
                        continue

                    w = kernel[kr, kc]
                    if np.abs(w) < min_w:
                        continue
                    
                    src = map_to_src(sr, sc, on_path, row_bits)
                    if src < 0 or src >=  num_src:
                        continue
                    # src = sr*layer_width + sc + src_start_idx
                    # divide values so that indices match the size of the
                    # Post (destination) next layer
                    
                    if dst >= num_dst:
                        continue
                        
                    src = np.uint32(src) 
                    dst = np.uint32(dst)
                    w = float(w);

                    if w < 0:
                        inh_conns.append((src, dst, w, inh_delay))
                        inh_counts[dst] += 1
                    elif w > 0:
                        exc_conns.append((src, dst, w, exc_delay))
                        exc_counts[dst] += 1
    
    if remove_inh_only:
        exc_conns[:], inh_conns[:] = remove_inh_only_dst(
                                        exc_conns, inh_conns, exc_counts)
                                                         
    
    return exc_conns, inh_conns



def remove_inh_only_dst(exc_conns, inh_conns, exc_counts):
    new_exc = exc_conns[:] #copy lists --- paranoid choice
    new_inh = inh_conns[:] #copy lists --- paranoid choice

    for i in range(len(exc_counts)):
        if exc_counts[i] == 0:
            new_exc[:] = [x for x in new_exc if x[1] != i]
            new_inh[:] = [x for x in new_inh if x[1] != i]

    return new_exc, new_inh



def inh_neighbours(r, c, row_step, col_step, kw, kh, correlation,
                   delay=1, selfdelay=4):
    if row_step >= kh or col_step >= kw:
        return []
    else:
        hlf_kw = kw//2
        hlf_kh = kh//2
        src = (r//row_step)*(imgw//col_step) + c//col_step
        
        nbr = []
        kr = 0
        for nr in range(r - hlf_kh, r + hlf_kh + 1, row_step):
            if nr < 0 or nr >= imgh:
                kr += 1
                continue
                
            kc = 0
            for nc in range(c - hlf_kw, c + hlf_kw + 1, col_step):
                if nc < 0 or nc >= imgw:
                    kc += 1
                    continue
                
                if nr == r and nc == c:
                    d = selfdelay
                    dst = src
                else:
                    d = delay
                    dst = (nr//row_step)*(imgw//col_step) + nc//col_step

                w = correlation[kr, kc]
                
                nbr.append( (src, dst, d, w) )
                
                kc += 1
            kr += 1
        
        return nbr


def lateral_inh_connector(layer_width, layer_height, correlation,
                          inh_weight, delay=1., self_delay=4,
                          col_step=1, row_step=1, 
                          col_start=0, row_start=0, min_w = 0.001):
    '''Assuming there's an interneuron layer with as many neurons as dest'''
    # exc_conns = []
    inh_conns = []
    kh, kw = correlation.shape
    half_kh, half_kw = kh//2, kw//2
    num_dst = ( (layer_height - row_start)//row_step )*\
              ( (layer_width  - col_start)//col_step )
              
    for dr in range(row_start, layer_height, row_step):
        for dc in range(col_start, layer_width, col_step):
            sr0 = dr - half_kh
            sc0 = dc - half_kw

            for kr in range(kh):
                sr = sr0 + kr
                if sr < 0 or sr >= layer_height:
                    continue

                for kc in range(kw):
                    sc = sc0 + kc
                    if sc < 0 or sc >= layer_width:
                        continue

                    #divide values so that indices match the size of the
                    #Post (destination) next layer
                    ### from Post to Interneurons
                    # src = (dr//row_step)*layer_width//col_step + (dc//col_step)
                    # exc_conns.append( (src, src, exc_weight, delay) )
                    
                    ### from Interneurons to Post
                    inh_conns += inh_neighbours(dr, dc, row_step, col_step, kw, kh,
                                                correlation, delay, selfdelay)
                    
    # return exc_conns, inh_conns
    return inh_conns
                        
                    

                    
