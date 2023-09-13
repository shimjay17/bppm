import cv2
import numpy as np
import os
import shutil
import warnings
warnings.filterwarnings('ignore')

from stitch.images import Image
from stitch.matching import MultiImageMatches, build_homographies, find_connected_components
from stitch.rendering import multi_band_blending, simple_blending, set_gain_compensations

import rectification.doctr_inference

from utils import unsharp_masking, get_logger, get_mask, crop_and_save, resize_save_image 

import ipdb

class ImageStitcherNode:
    def __init__(self, conf):
        self.conf = conf
        self.image_groups = conf.image_groups

        # Get logger
        self.logger = None 
        self.print = None 

        # Load DocTr model
        self.GeoTr_Seg_model = rectification.doctr_inference.GeoTr_Seg().cuda()
        rectification.doctr_inference.reload_segmodel(self.GeoTr_Seg_model.msk, self.conf.model.Seg_path)
        rectification.doctr_inference.reload_model(self.GeoTr_Seg_model.GeoTr, self.conf.model.GeoTr_path)
        
        # To eval mode
        self.GeoTr_Seg_model.eval()

    def produce_stitched_image(self, img_list, save_path):
        self.print('Stitch images...')
        images = []
        for i in range(len(img_list)): # for writting log.txt and storing each images
            self.print(f'input {i+1} : {img_list[i]}')
            images.append(Image(img_list[i]))
        self.print(f'output : {save_path}')

        if len(img_list) < 2: # if there is 1 or less images in the list (no images to stitch)
            shutil.copyfile(img_list[0], save_path) # save the image as result and quit the process. 
            return True
        
        # compute features 
        for image in images:
            image.compute_features()
        
        #match features of the images
        matcher = MultiImageMatches(images)
        pair_matches = matcher.get_pair_matches()
        pair_matches.sort(key=lambda pair_match: len(pair_match.matches), reverse=True) 

        #find connected componets
        connected_components = find_connected_components(pair_matches)

        #homographies 
        build_homographies(connected_components, pair_matches)

        #compute gain compensation
        cgc = True
        if cgc:
            for connected_component in connected_components:
                component_matches = [
                    pair_match for pair_match in pair_matches if pair_match.image_a in connected_components[0]
                ]

                set_gain_compensations(
                    connected_components[0],
                    component_matches,
                    sigma_n=10,
                    sigma_g=0.1,
                )      

            for image in images:
                image.image = (image.image * image.gain[np.newaxis, np.newaxis, :]).astype(np.uint8)

        results = []

        if self.conf.algorithm.stitcher.multiband_blending: #multi-band blending
            for connected_component in connected_components:
                results.append(multi_band_blending(connected_component, num_bands=5, sigma=1))
        else: #simple blending
            for connected_component in connected_components:
                results.append(simple_blending(connected_component))
        
        stitch_success = True
        if len(results) != 1:
            stitch_success = False
        
        if not stitch_success: # if stitching fails
            self.print('(Failed)')
            # return that stitching failed
            return False
        
        cv2.imwrite(save_path, results[0]) # save stitched images as result
        # return that stitching succeeded 
        return True 
    
    def stitch_one_group(self, group_images, save_dir):
        # make dict of cam_id keys, make dict of tilt keys containing corresponding images within the dict of cam_id dict. 
        group_dict = dict()
        group_dict_view = dict()
        for i in group_images:
            _, cam_id, tilt, pan, _ = os.path.basename(i).split("_") # split to get the info of each images
            if cam_id not in group_dict.keys(): # check the camera id of the image 
                cam_dir = os.path.join(save_dir, cam_id)
                os.makedirs(cam_dir, exist_ok=True)
                group_dict[cam_id] = dict() # make dictionary of cameras. key: camera, 
                group_dict_view[cam_id] = dict() # make dict of cameras. key: camera, for view
            if self.conf.algorithm.stitcher.stitch_vert_first: # if stitch vertical first, value: dictionary pan containing list of tilts corresponding to that pan
                if pan not in group_dict[cam_id].keys():
                    group_dict[cam_id][pan] = []
                group_dict[cam_id][pan].append(i)
            else: # if stitch horizontal first, value: dictionary of tilt containing list of pan 
                if tilt not in group_dict[cam_id].keys():
                    group_dict[cam_id][tilt] = []
                group_dict[cam_id][tilt].append(i)
            if self.conf.algorithm.view_for_user.stitch=='vert': # if view stitch vert, value: dictionary pan containing list of tilts corresponding to that pan
                if pan not in group_dict_view[cam_id].keys():
                    group_dict_view[cam_id][pan] = []
                group_dict_view[cam_id][pan].append(i)
            else: # if view stitch hori or all, value: dictionary of tilt containing list of pan 
                if tilt not in group_dict_view[cam_id].keys():
                    group_dict_view[cam_id][tilt] = []
                group_dict_view[cam_id][tilt].append(i)
        
        for cam_id , cam_d in group_dict.items(): # for printing pan/tilt image numbers for each camera
            num_secs_per_cams = {sec_id: len(secs) for sec_id, secs in cam_d.items()}
            self.print(f'Number of images for cam:{cam_id} = {num_secs_per_cams}')

        redo = False # if final and view do not align, we need to do it again.
        final_results = [] # final results directory 
        view_final_results = []
        stitched_cams = [] # stitched camera results directory
        cam_list_str = 'cam=[' # for naming stitched images. list of cameras
        sec_info_str = '[' # for naming stitched images. secondly stitched images info list
        for cam_k, cam in group_dict.items(): # stitch images of each camera           
            cam_list_str += cam_k+',' # update list of cameras for naming
            sec_list_str = 'tilt={' # for naming stitched images. when processing horizontal image first, stitch different pans first, so we will get stitched images of each tilts for the second stitch.
            if self.conf.algorithm.stitcher.stitch_vert_first: # when processing vertical image first, there will be stitched images of each pans.
                sec_list_str = 'pan={' 
            stitched_secs = [] # secondly stitched images directory list 
            fir_info_str = '{' # for naming stithced images. firstly stitched images info

            for sec_k, sec in cam.items(): #get dictionary of {key:secondly stitching tilt or pan info , value:list of firstly stitching pan or tilt info} in the camera.
                sec_list_str += sec_k+',' # update list of secondly stitched image for naming 
                fir_list_str = 'pan=(' # if horizontal stitching first, then we will stitch different pans first. this is naming of those pans of each tilts
                if self.conf.algorithm.stitcher.stitch_vert_first: # if vertical stitching first.
                    fir_list_str = 'tilt=('
                for s in sec: # sec is list of firstly stitching images path
                    gr, _, ti, pa, _ = os.path.basename(s).split('_') # get image info from the image name
                    if self.conf.algorithm.stitcher.stitch_vert_first: # depending on rather its horizontal or vertical stitching first, the stitched image name format is different.
                        fir_list_str += ti+','
                    else:
                        fir_list_str += pa+','
                fir_list_str = fir_list_str[:-1] + ')' # for naming. firstly stitched images info is saved as: ex. pan=(170,180,190)
                if self.conf.algorithm.stitcher.stitch_vert_first: #  for naming. if stitched vertical first, the stitched images of PANs are prepared for the second stitching. 
                    save_path = os.path.join(save_dir, cam_k, f'group={gr}_cam={cam_k}_PAN={sec_k}_info:{fir_list_str}.png') # we have stitched images of different PANs, so it is capitalized to make it clear
                    fir_info_str += sec_k+':'+fir_list_str[5:]+','
                else: # if stitched horizontal first, we have stitched images of different TILTs
                    save_path = os.path.join(save_dir, cam_k, f'group={gr}_cam={cam_k}_TILT={sec_k}_info:{fir_list_str}.png')
                    fir_info_str += sec_k+':'+fir_list_str[4:]+','

                #stitch images
                suc = self.produce_stitched_image(sec, save_path) # stitch images in the list of firstly stitching images. suc is bool of rather the stitching succeeded or not.
                
                if suc: # if stitching worked, 
                    self.print(f'Stitched image saved to {save_path}')
                    stitched_secs.append(save_path) # append to secondly stitched images directory list
                else: # if stitching failed,
                    for i in sec: # save each images used for stitching as final results.
                        final_results.append(i)
            sec_list_str = sec_list_str[:-1]+'}' # for naming
            fir_info_str = fir_info_str[:-1]+'}' # for naming
            
            if self.conf.algorithm.view_for_user.make_view:
                if (self.conf.algorithm.stitcher.stitch_vert_first == (self.conf.algorithm.view_for_user.stitch=='vert')) or (not self.conf.algorithm.stitcher.stitch_vert_first == (self.conf.algorithm.view_for_user.stitch=='hori')):
                    #stitch first stithced results to get the result for whole camera
                    for i in stitched_secs: # save first stitched results as final results.
                        view_final_results.append(i)
                else: 
                    redo = True
              
            sec_info_str += cam_k+':'+fir_info_str[:]+',' # for naming
            save_path = os.path.join(save_dir, cam_k, f'group={gr}_CAM={cam_k}_info:{sec_list_str}_{fir_info_str}.png') # stitched result of CAM, capitalized to make it clear
            suc = self.produce_stitched_image(stitched_secs, save_path) # list of firstly stitched images for second stitching.
            if suc: # if successful, 
                self.print(f'Stitched image saved to {save_path}')
                stitched_cams.append(save_path) # append to list of stitched camera results for group stitching 
            else: # if failed, 
                for i in stitched_secs: # first stitched images saved as final results.
                    final_results.append(i)

        cam_list_str = cam_list_str[:-1]+']' # for naming
        sec_info_str = sec_info_str[:-1]+']' # for naming

        for i in stitched_cams: # result of each camera is appended as final results.
            final_results.append(i)

        # if result and view does not allign 
        if redo:
            stitched_cams = [] # stitched camera results directory
            cam_list_str = 'cam=[' # for naming stitched images. list of cameras
            sec_info_str = '[' # for naming stitched images. secondly stitched images info list
            for cam_k, cam in group_dict_view.items(): # stitch images of each camera           
                cam_list_str += cam_k+',' # update list of cameras for naming
                sec_list_str = 'tilt={' # for naming stitched images. when processing horizontal image first, stitch different pans first, so we will get stitched images of each tilts for the second stitch.
                if self.conf.algorithm.view_for_user.stitch=='vert': # when processing vertical image first, there will be stitched images of each pans.
                    sec_list_str = 'pan={' 
                stitched_secs = [] # secondly stitched images directory list 
                fir_info_str = '{' # for naming stithced images. firstly stitched images info

                for sec_k, sec in cam.items(): #get dictionary of {key:secondly stitching tilt or pan info , value:list of firstly stitching pan or tilt info} in the camera.
                    sec_list_str += sec_k+',' # update list of secondly stitched image for naming 
                    fir_list_str = 'pan=(' # if horizontal stitching first, then we will stitch different pans first. this is naming of those pans of each tilts
                    if self.conf.algorithm.view_for_user.stitch=='vert': # if vertical stitching first.
                        fir_list_str = 'tilt=('
                    for s in sec: # sec is list of firstly stitching images path
                        gr, _, ti, pa, _ = os.path.basename(s).split('_') # get image info from the image name
                        if self.conf.algorithm.view_for_user.stitch=='vert': # depending on rather its horizontal or vertical stitching first, the stitched image name format is different.
                            fir_list_str += ti+','
                        else:
                            fir_list_str += pa+','
                    fir_list_str = fir_list_str[:-1] + ')' # for naming. firstly stitched images info is saved as: ex. pan=(170,180,190)
                    if self.conf.algorithm.view_for_user.stitch=='vert': #  for naming. if stitched vertical first, the stitched images of PANs are prepared for the second stitching. 
                        save_path = os.path.join(save_dir, cam_k, f'group={gr}_cam={cam_k}_PAN={sec_k}_info:{fir_list_str}.png') # we have stitched images of different PANs, so it is capitalized to make it clear
                        fir_info_str += sec_k+':'+fir_list_str[5:]+','
                    else: # if stitched horizontal first, we have stitched images of different TILTs
                        save_path = os.path.join(save_dir, cam_k, f'group={gr}_cam={cam_k}_TILT={sec_k}_info:{fir_list_str}.png')
                        fir_info_str += sec_k+':'+fir_list_str[4:]+','

                    #stitch images
                    suc = self.produce_stitched_image(sec, save_path) # stitch images in the list of firstly stitching images. suc is bool of rather the stitching succeeded or not.
                    
                    if suc: # if stitching worked, 
                        self.print(f'Stitched image saved to {save_path}')
                        stitched_secs.append(save_path) # append to secondly stitched images directory list
                    else: # if stitching failed,
                        for i in sec: # save each images used for stitching as final results.
                            view_final_results.append(i)
                sec_list_str = sec_list_str[:-1]+'}' # for naming
                fir_info_str = fir_info_str[:-1]+'}' # for naming

                for i in stitched_secs: # save first stitched results as final results.
                    view_final_results.append(i)

                sec_info_str += cam_k+':'+fir_info_str[:]+',' # for naming
                save_path = os.path.join(save_dir, cam_k, f'group={gr}_CAM={cam_k}_info:{sec_list_str}_{fir_info_str}.png') # stitched result of CAM, capitalized to make it clear
                suc = self.produce_stitched_image(stitched_secs, save_path) # list of firstly stitched images for second stitching.
                if suc: # if successful, 
                    self.print(f'Stitched image saved to {save_path}')
                    stitched_cams.append(save_path) # append to list of stitched camera results for group stitching 
                else: # if failed, 
                    for i in stitched_secs: # first stitched images saved as final results.
                        view_final_results.append(i)

            cam_list_str = cam_list_str[:-1]+']' # for naming
            sec_info_str = sec_info_str[:-1]+']' # for naming

        fr_path_list = []
        for i in final_results: # list of final results. call each one
            n = os.path.basename(i)
            s_path = os.path.join(save_dir, n)
            s_img = cv2.imread(i)

            cv2.imwrite(s_path, unsharp_masking(s_img)) # save final results to the result directory
            #save final result in final_results folder
            fr_img = cv2.imread(s_path)
            fr_path = os.path.join(self.conf.log_dir,'final_results',n)
            fr_path_list.append(fr_path)
            cv2.imwrite(fr_path, fr_img)
            self.print(f'Final result saved to {fr_path}')

            
        processed = False
        if self.conf.algorithm.view_for_user.make_view: # if make view images,
            if self.conf.algorithm.view_for_user.stitch=='all': #if stitch all for view
                view_final_results = fr_path_list
            else:
                for i in view_final_results:
                    s_img = cv2.imread(i)
                    cv2.imwrite(i, unsharp_masking(s_img))

            for i in view_final_results:
                n = os.path.basename(i)
                self.print(f'--- For result {n} ---')
                s_path = i
  
              
                dir_name = os.path.join(os.path.dirname(os.path.dirname(s_path)), 'out_images')
                if self.conf.algorithm.view_for_user.stitch != 'all':
                    dir_name = os.path.dirname(os.path.dirname(os.path.dirname(s_path)))
                if self.conf.algorithm.rectangling.apply_rectangling: 
                    if self.conf.algorithm.rectangling.warp_instead_of_crop:
                        # Apply warp-based rectangling
                        self.print('Apply rectangling through warping...')
                        rect_save_path = os.path.join(dir_name,'warp_rect_' + os.path.basename(s_path))
                        rectification.doctr_inference.rec(s_path, rect_save_path, self.GeoTr_Seg_model)
                        s_path = rect_save_path
                        self.print(f'Rectangling done, saved to {rect_save_path}')
                        processed = True
                    else:
                        # Apply crop-based rectangling
                        self.print('Apply rectangling through cropping...')
                        rect_save_path = os.path.join(dir_name, 'crop_rect_' + os.path.basename(s_path))
                        mask = get_mask(s_path)
                        _ = crop_and_save(s_path, mask, rect_save_path, self.conf.algorithm.rectangling.crop_frac)
                        s_path = rect_save_path
                        self.print(f'Rectangling done, saved to {rect_save_path}')
                        processed = True

                if self.conf.algorithm.postprocessing.apply_postprocessing:
                    # Apply postprocessing
                    self.print('Apply postprocessing...')
                    pp_save_path = os.path.join(dir_name, 'pp_'+os.path.basename(s_path))
                    pp_img = self.postprocessing(s_path)
                    cv2.imwrite(pp_save_path , pp_img)
                    s_path = pp_save_path
                    self.print(f'Postprocessing done, saved to {pp_save_path}')
                    processed = True

                if self.conf.data.resize_output:
                    # Resize final result
                    self.print('Resize ..')
                    rs_save_path = os.path.join(dir_name, 'rs_'+os.path.basename(s_path))
                    resize_save_image(s_path, rs_save_path, tuple(map(int, self.conf.data.out_resolution.split(',')))) 

                    s_path = rs_save_path
                    self.print(f'Resizing done, saved to {rs_save_path}')
                    processed = True
                
                view_img = cv2.imread(s_path)

                view_path = os.path.join(os.path.dirname(os.path.dirname(s_path)),'views',os.path.basename(s_path))
                cv2.imwrite(view_path , view_img)
                
            if not processed:
                self.print('no postprocess done')

        self.print(os.path.join(self.conf.log_dir,'final_results'))


    def stitch_all(self):
        out_dir = os.path.join(self.conf.log_dir, 'out_images')
        for group_id, group_images in self.image_groups.items():
            save_dir = os.path.join(out_dir, group_id) 
            os.makedirs(save_dir, exist_ok=True) # make out_images directory
            logger = get_logger(os.path.join(self.conf.log_dir, f'log_{group_id}.txt')) #make log for each group
            self.logger = logger
            self.print = self.logger.info
            self.print(f'------------ Stitch Group {group_id} ------------')
            if self.conf.algorithm.stitcher.stitch_vert_first: # check stitcher config for print.
                sti = 'vertically'
            else:
                sti = 'horizontally'
            self.print(f'Stitch {sti} first')
            if self.conf.algorithm.rectangling.apply_rectangling:
                if self.conf.algorithm.rectangling.warp_instead_of_crop:
                    self.print('Apply rectangling through rectification(warp)')
                else:
                    self.print('Apply rectangling through crop')
            if self.conf.algorithm.postprocessing.apply_postprocessing:
                self.print('Apply postprocessing')

            self.stitch_one_group(group_images, save_dir) # stitch one group at a time.

            while logger.hasHandlers(): # end the log
                logger.removeHandler(logger.handlers[0])

    def postprocessing(self, save_path):
        brightness_ratio = self.conf.algorithm.postprocessing.brightness_ratio
        brightness_pixel = self.conf.algorithm.postprocessing.brightness_pixel
        saturation_ratio = self.conf.algorithm.postprocessing.saturation_ratio
        saturation_pixel = self.conf.algorithm.postprocessing.saturation_pixel       
        
        image = cv2.imread(save_path)
        mask = get_mask(save_path)
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
              
        #brightness
        if brightness_ratio == 1.0 and brightness_pixel == 0:
            self.print('Brightness: no adjustment')
        else:
            self.print(f'Brightness: Ratio: {brightness_ratio}, Pixel: {brightness_pixel} adjusted')
            image = image.astype('uint8')
            image = np.where(mask_3d==255, cv2.convertScaleAbs(image, alpha=brightness_ratio, beta=brightness_pixel), image)

        #saturation
        if saturation_ratio == 1.0 and saturation_pixel == 0:
            self.print('Saturation: no adjustment')
        else:
            self.print(f'Saturation: Ratio: {saturation_ratio}, Pixel: {saturation_pixel} adjusted')
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            adj_sat = np.clip(hsv[:, :, 1] * saturation_ratio + saturation_pixel, 0, 255)
            hsv[:, :, 1] = np.where(mask_3d[:, :, 0]==255, hsv[:, :, 1], adj_sat)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image
