import numpy as np
import cv2
import pygame
import os
import carla
import math

from PIL import Image
from constants import LABEL_COLORS
from bbox import BoundingBox, Filtering, Transform

class SensorProcessor:
    @staticmethod
    def process_rgb(sensor_data, is_thermal=False):
        array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
        array = array[:, :, :3]
        
        if is_thermal:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            if array.ndim == 2:  
                array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        else:
            array = array
        return array

    # @staticmethod
    # def process_depth(sensor_data):
    #     array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
    #     array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
    #     array = array[:, :, :3]
        
    #     R = array[:, :, 2].astype(np.float32)
    #     G = array[:, :, 1].astype(np.float32)
    #     B = array[:, :, 0].astype(np.float32)
        
    #     depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
    #     depth = depth * 1000.0 # meter 
    #     depth = np.clip(depth, 0, 65535)
    #     return depth
    @staticmethod
    def process_depth(raw_data):
        """
        CARLA Depth 이미지 올바른 처리
        CARLA는 RGB 3채널에 depth를 인코딩함
        """
        # Raw 데이터를 numpy array로 변환
        array = np.frombuffer(raw_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (raw_data.height, raw_data.width, 4))  # BGRA
        array = array[:, :, :3]  # RGB만 사용
        array = array[:, :, ::-1]  # BGR → RGB 변환
        
        # CARLA 공식 depth 디코딩 공식 적용
        # normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        # in_meters = 1000 * normalized
        
        R = array[:, :, 0].astype(np.float32)
        G = array[:, :, 1].astype(np.float32) 
        B = array[:, :, 2].astype(np.float32)
        
        # 24-bit depth 값 복원
        normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
        
        # meter로 변환 (CARLA의 far plane = 1000m)
        depth_meters = 1000.0 * normalized
                
        return depth_meters


    @staticmethod
    def process_depth_for_visualization(depth_array):
        """
        Raw depth array를 시각화용으로 변환
        
        Parameters:
        - depth_array: 원본 깊이 배열 (mm 단위의 float32)
        
        Returns:
        - visualization_depth: 시각화용 8비트 이미지
        """
        # depth_array는 이미 mm 단위로 저장된 float32 배열
        # 시각화를 위해 0-255 범위로 정규화
        # 일반적인 깊이 범위(0-100m)를 0-255로 스케일링
        normalized_depth = np.clip(depth_array / 100.0, 0, 255).astype(np.uint8)
        return normalized_depth

    @staticmethod
    def process_segmentation(sensor_data):
        array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
        array = array[:, :, 2]  # 원본 label
        return array

class DisplayImage:
    @staticmethod
    def get_display_image(sensor):
        if not sensor or not sensor.get('raw_data'):
            return None
            
        if sensor['type'] == "sensor.camera.rgb":
            array = SensorProcessor.process_rgb(sensor['raw_data'], 
                                            is_thermal=(sensor['id'] == "keti_camera9_thermal"))
            display_image = array.copy()

            all_bboxes = []
            for bbox_data in sensor.get('bbox_3d_coords', []):
                if bbox_data is not None:
                    all_bboxes.append(bbox_data)
            for static_data in sensor.get('static_infos', []):
                if static_data is not None:
                    all_bboxes.append(static_data)
            
            filtered_bboxes = all_bboxes

            # Draw bounding boxes if available
            if hasattr(sensor['sensor_spawned'], 'calibration'):
                
                for bbox_data in filtered_bboxes:
                    if bbox_data is not None:
                        if len(bbox_data) >= 2:
                            class_id, vertices_2d = bbox_data[:2]  # 처음 2개만 사용
                            try:
                                display_image = BoundingBox.draw_3d_bbox(display_image, vertices_2d, class_id)
                            except Exception as e:
                                print(f"Error drawing bbox: {str(e)}")
                                import traceback
                                traceback.print_exc()
                
                # bbox viz 저장
                camera_id = sensor['id']
                viz_bbox_dir = os.path.join(sensor.get('save_path', ''), 'viz_bbox')
                os.makedirs(viz_bbox_dir, exist_ok=True)
                
                frame_id = sensor.get('frame_id', 0)
                
                viz_bbox_path = os.path.join(viz_bbox_dir, f'{frame_id:08d}.png')
                cv2.imwrite(viz_bbox_path, display_image)

            # 원본 -> sensor_data
            sensor['original_image'] = array
            return display_image

        elif sensor['type'] == "sensor.camera.depth":
            depth = SensorProcessor.process_depth(sensor['raw_data'])
            visualization_depth = SensorProcessor.process_depth_for_visualization(depth)
            
            return cv2.cvtColor(visualization_depth, cv2.COLOR_GRAY2BGR)

        elif sensor['type'] == "sensor.camera.semantic_segmentation":
            array = SensorProcessor.process_segmentation(sensor['raw_data'])
            color_array = LABEL_COLORS[array]
            return (color_array * 255).astype(np.uint8)
            
        elif sensor['type'] == "sensor.lidar.ray_cast":
            dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
            return dummy_img
            
        return None

def draw_cell(canvas, image, row, col, cell_height, cell_width):
    if image is not None:
        resized = cv2.resize(image, (cell_width, cell_height))
        y_start = row * cell_height
        x_start = col * cell_width
        y_end = min(y_start + cell_height, canvas.shape[0])
        x_end = min(x_start + cell_width, canvas.shape[1])
        canvas[y_start:y_end, x_start:x_end] = resized[:(y_end-y_start), :(x_end-x_start)]


def draw_images(surface, data_dict, save_dir, frame):
    cameras = {
        'rgb': [],
        'depth': [],
        'segmentation': [],
        'lidar': []
    }
    
    sensor_info = {
        'rgb': [],
        'depth': [],
        'segmentation': [],
        'lidar': []
    }

    for sensor in data_dict:
        sensor['frame_id'] = frame
        
        if sensor['type'].startswith('sensor.camera'):
            sensor_type = sensor['type'].split('.')[-1]
            if sensor_type == 'semantic_segmentation':
                sensor_type = 'segmentation'
            if sensor_type in cameras:
                display_image = DisplayImage.get_display_image(sensor)
                if display_image is not None:
                    cameras[sensor_type].append({
                        'image': display_image,
                        'sensor_data': sensor,
                        'bbox_3d_coords': sensor.get('bbox_3d_coords', []),
                        'camera': sensor['sensor_spawned']
                    })
                    
                    sensor_info[sensor_type].append({
                        'image': display_image.copy(),
                        'sensor_data': sensor,
                        'camera': sensor['sensor_spawned']
                    })

        elif sensor['type'].startswith('sensor.lidar'):
            display_image = DisplayImage.get_display_image(sensor)
            if display_image is not None:
                cameras['lidar'].append({
                    'image': display_image,
                    'sensor_data': sensor,
                    'camera': None
                })
                
                sensor_info['lidar'].append({
                    'image': display_image,
                    'sensor_data': sensor,
                    'camera': None
                })

    active_types = [typ for typ, images in cameras.items() if len(cameras[typ]) > 0]
    
    #  (1920x1080)
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    num_rows = len(active_types)
    max_cols = max([len(cameras[typ]) for typ in active_types]) if active_types else 0
    
    if num_rows == 0 or max_cols == 0:
        return

    cell_height = 1080 // num_rows
    cell_width = 1920 // max_cols

    for row, sensor_type in enumerate(active_types):
        for col, sensor_data in enumerate(cameras[sensor_type]):
            if col < max_cols:
                image = sensor_data['image']
                
                if sensor_type == 'rgb':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                draw_cell(canvas, image, row, col, cell_height, cell_width)

    display_image = cv2.resize(canvas, (surface.get_width(), surface.get_height()))
    surf = pygame.surfarray.make_surface(display_image.swapaxes(0, 1))
    surface.blit(surf, (0, 0))

    merged_dir = os.path.join(save_dir, 'merged_image')
    os.makedirs(merged_dir, exist_ok=True)
    merged_path = os.path.join(merged_dir, f'{frame:08d}.png')
    
    canvas= cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    cv2.imwrite(merged_path, canvas)
    
    del canvas
    del surf