import numpy as np
import carla
import cv2
import math
import json
import uuid
import time
import os

import transforms3d.euler
import transforms3d.quaternions

from datetime import datetime
from lidar import MultiOT128Processor
from constants import LABEL_COLORS


class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}  # actor_id -> track_info
        self.next_track_id = 1
        
    def update_objects(self, current_objects):
        """
        current_objects: [(actor_id, class_id, center_3d)] , type : list
        Returns: [(actor_id, track_id, class_id)] , type : list
        """
        current_frame_objects = {}
        
        for actor_id, class_id, center_3d in current_objects:
            if actor_id in self.tracked_objects:
                track_info = self.tracked_objects[actor_id]
                track_info['last_seen'] = time.time()
                track_info['position'] = center_3d
                current_frame_objects[actor_id] = track_info
            else:
                track_info = {
                    'track_id': self.next_track_id,
                    'class_id': class_id,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'position': center_3d
                }
                self.tracked_objects[actor_id] = track_info
                current_frame_objects[actor_id] = track_info
                self.next_track_id += 1
        
        # 일정 시간 보이지 않은 객체 제거 (10초)
        current_time = time.time()
        to_remove = []
        for actor_id, track_info in self.tracked_objects.items():
            if current_time - track_info['last_seen'] > 10.0:
                to_remove.append(actor_id)
        
        for actor_id in to_remove:
            del self.tracked_objects[actor_id]
        
        # 결과 반환
        result = []
        for actor_id, track_info in current_frame_objects.items():
            result.append((actor_id, track_info['track_id'], track_info['class_id']))
        
        return result

class TrackUUIDManager:
    """TrackId -> UUID"""
    def __init__(self):
        self.track_id_to_uuid = {}    # track_id -> UUID 매핑
    
    def get_track_uuid(self, track_id):
        if track_id not in self.track_id_to_uuid:
            self.track_id_to_uuid[track_id] = str(uuid.uuid4())
        return self.track_id_to_uuid[track_id]

class BoundingBox:
    @staticmethod
    def get_lidar_based_3d_bboxes(world, ego_vehicle_id, static_vehicles, object_tracker, max_distance=150.0, combined_points=None):
        ego_vehicle = world.get_actor(ego_vehicle_id)
        if not ego_vehicle:
            return []
            
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        
        bbox_list = []
        static_bbox_list = []  # Static vehicle
        
        # 1. Dynamic vehicles
        vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if vehicle.id == ego_vehicle_id:
                continue
                
            bbox_data = BoundingBox._extract_3d_bbox_data(vehicle, is_static=False)
            if bbox_data:
                bbox_list.append(bbox_data)
        
        # 2. Pedestrians
        pedestrians = world.get_actors().filter('walker.pedestrian.*')
        for pedestrian in pedestrians:
            # Distance filtering
            distance = pedestrian.get_transform().location.distance(ego_location)
            if distance > max_distance:
                continue
                
            bbox_data = BoundingBox._extract_3d_bbox_data(pedestrian, is_static=False)
            if bbox_data:
                bbox_list.append(bbox_data)
        
        # 3. Static vehicles 
        for static_vehicle in static_vehicles:
            if not static_vehicle.get('is_map_static', False):
                continue
                                
            bbox_data = BoundingBox._extract_static_3d_bbox_data(static_vehicle)
            if bbox_data:
                static_bbox_list.append(bbox_data)
        
        # 4. Static vehicles -> OBB Filtering
        if static_bbox_list:
            filtered_static_bboxes = Filtering.filter_static_vehicles_obb(static_bbox_list)
            bbox_list.extend(filtered_static_bboxes)
        
        # Object tracking update
        tracking_input = [(bbox['actor_id'], bbox['class_id'], bbox['center_3d']) for bbox in bbox_list]
        tracked_objects = object_tracker.update_objects(tracking_input)
        
        # Add tracking info to bbox data
        track_dict = {actor_id: (track_id, class_id) for actor_id, track_id, class_id in tracked_objects}
        for bbox in bbox_list:
            if bbox['actor_id'] in track_dict:
                bbox['track_id'] = track_dict[bbox['actor_id']][0]
        
        return bbox_list
    
    @staticmethod
    def _extract_3d_bbox_data(actor, is_static=False):
        """Extract 3D bounding box data from dynamic actor with semantic tag"""
        try:
            # Get current transform and bounding box
            transform = actor.get_transform()
            bbox = actor.bounding_box
            
            # Get world vertices
            vertices_world = bbox.get_world_vertices(transform)
            
            # Calculate center in world coordinates
            center_world = transform.transform(bbox.location)
            center_3d = [float(center_world.x), float(center_world.y), float(center_world.z)]
            
            # Size (extent * 2)
            extent = bbox.extent
            size_3d = [float(extent.x * 2), float(extent.y * 2), float(extent.z * 2)]
            
            # Determine class
            class_id = 0  # default: vehicle
            if actor.type_id.startswith('walker'):
                class_id = 2  # pedestrian
            elif 'number_of_wheels' in actor.attributes:
                wheels = int(actor.attributes['number_of_wheels'])
                class_id = 1 if wheels == 2 else 0
            
            # Extract semantic tag
            semantic_tag = Filtering.get_actor_semantic_tag(actor)
            
            return {
                'actor_id': actor.id,
                'class_id': class_id,
                'semantic_tag': semantic_tag,  
                'center_3d': center_3d,
                'size_3d': size_3d,
                'yaw': float(transform.rotation.yaw),
                'vertices_world': vertices_world,
                'bbox_location': bbox.location,
                'bbox_extent': bbox.extent,
                'transform': transform,
                'is_static': is_static
            }
            
        except Exception as e:
            print(f"Error extracting bbox data from actor {actor.id}: {e}")
            return None

    
    @staticmethod
    def _extract_static_3d_bbox_data(static_vehicle):
        """Extract 3D bounding box data from static vehicle"""
        try:
            bbox = static_vehicle['bounding_box']
            
            # For static vehicles, get_level_bbs returns world coordinates already
            vertices_world = static_vehicle['vertices']
            
            # Center in world coordinates
            center_3d = [float(bbox.location.x), float(bbox.location.y), float(bbox.location.z)]
            
            # Size
            extent = bbox.extent
            size_3d = [float(extent.x * 2), float(extent.y * 2), float(extent.z * 2)]
            
            # Class determination
            vehicle_type = static_vehicle['type']
            class_id = 1 if vehicle_type in [int(carla.CityObjectLabel.Motorcycle), 
                                            int(carla.CityObjectLabel.Bicycle)] else 0
           # Semantic Tag - collect_static_vehicles
            semantic_tag = static_vehicle['type'] 

            return {
                'actor_id': static_vehicle['id'],
                'class_id': class_id,
                'center_3d': center_3d,
                'semantic_tag': semantic_tag,
                'size_3d': size_3d,
                'yaw': float(bbox.rotation.yaw),
                'vertices_world': vertices_world,
                'bbox_location': bbox.location,
                'bbox_extent': bbox.extent,
                'transform': carla.Transform(bbox.location, bbox.rotation),
                'is_static': True
            }
            
        except Exception as e:
            print(f"Error extracting static bbox data: {e}")
            return None        
        
    @staticmethod
    def get_environment_vehicle_bbox(env_object, camera_sensor, depth_data=None, occluded_check=False, coord="world"):
        """Static bbox for RGB Camera"""
        if 'vertices' not in env_object or not env_object['vertices']:
            return None
            
        if not hasattr(camera_sensor, 'calibration'):
            return None
        K = np.array(camera_sensor.calibration)
        
        world_2_camera = np.array(camera_sensor.get_transform().get_inverse_matrix())
        
        if hasattr(env_object, 'transform'):
            object_location = env_object['transform'].location
        elif 'transform' in env_object:
            object_location = env_object['transform'].location
        else:
            if 'bounding_box' in env_object:
                object_location = env_object['bounding_box'].location
            else:
                return None
                
        if not Filtering.filter_by_camera_view(object_location, camera_sensor):
            return None

        vertices3d = env_object['vertices']
     
        vertices_2d = []
        for vertex in vertices3d:
            point_2d = Transform.get_image_point(vertex, K, world_2_camera)
            if point_2d is not None:
                vertices_2d.append(point_2d)
        
        if len(vertices_2d) < 4:
            return None
            
        width = int(camera_sensor.attributes.get('image_size_x', 1920))
        height = int(camera_sensor.attributes.get('image_size_y', 1080))
         
        vertices_2d_array = np.array(vertices_2d)
        if not Filtering.check_points_in_screen(vertices_2d_array, width, height):
            return None
            
        # Class ID
        class_id = 1 if env_object['type'] in [int(carla.CityObjectLabel.Motorcycle), 
                                            int(carla.CityObjectLabel.Bicycle)] else 0
        bbox = env_object['bounding_box']
        actor_id = env_object.get('id', f"static_{class_id}")
        
        return class_id, vertices_2d_array, actor_id, bbox.location, bbox.extent, bbox.rotation.yaw, coord
    
    @staticmethod
    def get_bounding_box(object, ego_vehicle_id, sensor, depth_data=None, occluded_check=False, coord="world"):
        """Dynamic BBOX for RGB Camera"""
        if object.id == ego_vehicle_id:
            return None

        try:
            transform = object.get_transform()
            
            if not Filtering.filter_by_camera_view(transform.location, sensor):
                return None
                
        except Exception as e:
            return None

        world_2_camera = np.array(sensor.get_transform().get_inverse_matrix())
        if not hasattr(sensor, 'calibration'):
            return None
        
        K = np.array(sensor.calibration)

        bbox = object.bounding_box
        current_transform = object.get_transform()
        
        if coord == "local":
            local_vs = bbox.get_local_vertices()
            vertices3d = [transform.transform(v) for v in local_vs]
        else:  # world
            vertices3d = bbox.get_world_vertices(current_transform)

        vertices_2d = []
        for vertex in vertices3d:
            point_2d = Transform.get_image_point(vertex, K, world_2_camera)
            if point_2d is None:
                return None
            vertices_2d.append(point_2d)

        if len(vertices_2d) != 8: 
            return None

        vertices_2d = np.array(vertices_2d)
        
        width = int(sensor.attributes.get('image_size_x', 1920))
        height = int(sensor.attributes.get('image_size_y', 1080))
        
        if not Filtering.check_points_in_screen(vertices_2d, width, height, min_visible_points=4):
            return None

        class_id = 0  # 기본: vehicle
        if object.type_id.startswith('walker'):
            class_id = 2
        elif 'number_of_wheels' in object.attributes:
            wheels = int(object.attributes['number_of_wheels'])
            class_id = 1 if wheels == 2 else 0
        
        actor_id = object.id
        return class_id, vertices_2d, actor_id, bbox.location, bbox.extent, transform.rotation.yaw, coord
    
    @staticmethod
    def draw_3d_bbox(image, vertices_2d, class_id=0):
        """3D BBOX for RGB Camera """
        edges = [[0,1], [1,3], [3,2], [2,0],  # bottom
                [4,5], [5,7], [7,6], [6,4],  # top
                [0,4], [1,5], [2,6], [3,7]]  # sides
        
        colors = {
            0: (255, 0, 0),    # Vehicle - Red
            1: (0, 255, 0),    # Bicycle - Green
            2: (0, 0, 255)     # Pedestrian - Blue
        }
        color = colors.get(class_id, (255, 255, 255))

        for edge in edges:
            try:
                p1 = vertices_2d[edge[0]]
                p2 = vertices_2d[edge[1]]
                cv2.line(image,
                        (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])),
                        color, 2)
            except Exception as e:
                print(f"Error drawing edge {edge}: {e}")
        
        return image

class Filtering:
    ### DETPH FILTERING
    @staticmethod
    def depth_based_validation(bbox, camera_sensors, depth_data_dict, min_validation_ratio=0.3):
        if not camera_sensors or not depth_data_dict:
            return True  
        
        validation_result = Transform.get_bbox_camera_validation_score(
            bbox, camera_sensors, depth_data_dict, min_validation_ratio
        )
        
        actor_id = bbox.get('actor_id', 'unknown')
        class_id = bbox.get('class_id', 0)
        is_static = bbox.get('is_static', False)
        
        object_type = "static" if is_static else ("pedestrian" if class_id == 2 else "dynamic")
        
        return validation_result['valid']

    @staticmethod
    def depth_based_validation(bbox, camera_sensors, depth_data_dict, min_validation_ratio=0.2):
        if not camera_sensors or not depth_data_dict:
            return True 
        
        validation_result = Transform.get_bbox_camera_validation_score(
            bbox, camera_sensors, depth_data_dict, min_validation_ratio
        )
        
        actor_id = bbox.get('actor_id', 'unknown')
        class_id = bbox.get('class_id', 0)
        is_static = bbox.get('is_static', False)
        
        object_type = "static" if is_static else ("pedestrian" if class_id == 2 else "dynamic")
        
        return validation_result['valid']

    @staticmethod
    def filter_bboxes_by_semantic_and_depth(bboxes, semantic_detections, camera_sensors=None, depth_data_dict=None):
        if not bboxes:
            return []

        filtered_bboxes = []
        stats = {
            'dynamic_semantic_kept': 0,
            'dynamic_semantic_filtered': 0,
            'pedestrian_depth_kept': 0,
            'pedestrian_depth_filtered': 0,
            'static_depth_kept': 0,
            'static_depth_filtered': 0,
            'other_kept': 0
        }
        
        # Semantic detection에서 dynamic actor IDs
        detected_actor_ids = set(semantic_detections.keys()) if semantic_detections else set()
        
        for i, bbox in enumerate(bboxes):
            actor_id = bbox.get('actor_id')
            is_static = bbox.get('is_static', False)
            class_id = bbox.get('class_id', 0)
            
            # 1. Dynamic Vehicle (class_id=0, is_static=False)
            if not is_static and class_id == 0:  # Dynamic Vehicle
                if actor_id in detected_actor_ids:
                    filtered_bboxes.append(bbox)
                    stats['dynamic_semantic_kept'] += 1
                else:
                    stats['dynamic_semantic_filtered'] += 1
            
            # 2. Pedestrian (class_id=2, is_static=False)  
            elif not is_static and class_id == 2:  # Pedestrian
                if camera_sensors and depth_data_dict:
                    depth_valid = Filtering.depth_based_validation(bbox, camera_sensors, depth_data_dict)
                    if depth_valid:
                        filtered_bboxes.append(bbox)
                        stats['pedestrian_depth_kept'] += 1
                    else:
                        stats['pedestrian_depth_filtered'] += 1
                else:
                    filtered_bboxes.append(bbox)
                    stats['pedestrian_depth_kept'] += 1
            
            # 3. Static Vehicle (is_static=True)
            elif is_static:  # Static Vehicle
                if not camera_sensors or not depth_data_dict:
                    filtered_bboxes.append(bbox)
                    stats['static_depth_kept'] += 1
                else:
                    # 10% 카메라만 통과하면 통과 ㄱㄱ
                    validation_result = Transform.get_bbox_camera_validation_score(
                        bbox, camera_sensors, depth_data_dict, min_validation_ratio=0.3
                    )
                    
                    if validation_result['valid'] or validation_result['validated_cameras'] > 0:
                        filtered_bboxes.append(bbox)
                        stats['static_depth_kept'] += 1
                    else:
                        # 그래도 실패하면 거리로 ㄱ
                        center_3d = bbox.get('center_3d', [0, 0, 0])
                        distance = np.linalg.norm(center_3d)
                                    
            # 4. 기타 Obj (Bicycle 등)
            else:
                filtered_bboxes.append(bbox)
                stats['other_kept'] += 1
        
        return filtered_bboxes
    
    ### Semantic Filtering
    @staticmethod
    def get_actor_semantic_tag(actor):
        """Export CARLA Dynamic Actor ~ Semantic Tag """
        
        # 1. Vehicle Actor
        if actor.type_id.startswith('vehicle.'):
            if 'motorcycle' in actor.type_id.lower():
                return int(carla.CityObjectLabel.Motorcycle)
            elif 'bicycle' in actor.type_id.lower():
                return int(carla.CityObjectLabel.Bicycle)
            elif 'truck' in actor.type_id.lower():
                return int(carla.CityObjectLabel.Truck)
            elif 'bus' in actor.type_id.lower():
                return int(carla.CityObjectLabel.Bus)
            else:
                return int(carla.CityObjectLabel.Car)
        
        # 2. Pedestrian Actor
        elif actor.type_id.startswith('walker.pedestrian'):
            return int(carla.CityObjectLabel.Pedestrians)
        
        else:
            return int(carla.CityObjectLabel.Other)
        
    @staticmethod
    def filter_bboxes_by_semantic_lidar(bboxes, semantic_detections):
        """
        Semantic LiDAR 결과로 BBox 필터링
        - Dynamic: object_idx (Actor ID) 
        - Static: object_tag (Semantic Tag) 
        
        Args:
            bboxes: BBox 리스트  
            semantic_detections: process_semantic_lidar_data 결과 {object_idx: object_tag}
        
        Returns:
            필터링된 BBox 리스트
        """
        if not semantic_detections:
            print("No semantic detections available - keeping all bboxes")
            return bboxes
        
        # Dynamic: Actor ID 
        detected_actor_ids = set(semantic_detections.keys())
        
        # Static: Semantic Tag 
        detected_semantic_tags = set(semantic_detections.values())

        filtered_bboxes = []
        filter_stats = {
            'kept_dynamic': 0, 'kept_static': 0,
            'filtered_dynamic': 0, 'filtered_static': 0
        }
        
        for bbox in bboxes:
            actor_id = bbox.get('actor_id')
            is_static = bbox.get('is_static', False)
            bbox_semantic_tag = bbox.get('semantic_tag')
            
            if is_static:
                if bbox_semantic_tag in detected_semantic_tags:
                    filtered_bboxes.append(bbox)
                    filter_stats['kept_static'] += 1
                else:
                    filter_stats['filtered_static'] += 1
            else:
                if actor_id in detected_actor_ids:
                    filtered_bboxes.append(bbox)
                    filter_stats['kept_dynamic'] += 1
                    is_pedestrian = bbox.get('class_id') == 2
                    actor_type = "pedestrian" if is_pedestrian else "vehicle"
                else:
                    filter_stats['filtered_dynamic'] += 1
                    is_pedestrian = bbox.get('class_id') == 2
                    actor_type = "pedestrian" if is_pedestrian else "vehicle"
                
        return filtered_bboxes        

    @staticmethod
    def process_frame_with_semantic_filtering(world, ego_vehicle_id, static_vehicles, 
                                            semantic_lidar_data, object_tracker):
        
        # 1. BBox 생성 (Semantic Tag 포함)
        bbox_list = []
        
        # Dynamic vehicles
        vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if vehicle.id == ego_vehicle_id:
                continue
                
            bbox_data = BoundingBox._extract_3d_bbox_data(vehicle, is_static=False)
            if bbox_data:
                bbox_list.append(bbox_data)
        
        # Pedestrians  
        pedestrians = world.get_actors().filter('walker.pedestrian.*')
        for pedestrian in pedestrians:
            bbox_data = BoundingBox._extract_3d_bbox_data(pedestrian, is_static=False)
            if bbox_data:
                bbox_list.append(bbox_data)
        
        # Static vehicles
        for static_vehicle in static_vehicles:
            bbox_data = BoundingBox._extract_static_3d_bbox_data(static_vehicle)
            if bbox_data:
                bbox_list.append(bbox_data)
        
        # 2. Semantic LiDAR 데이터 처리
        semantic_detections = MultiOT128Processor.process_semantic_lidar_data(semantic_lidar_data)
        
        # 3. Semantic 필터링 적용
        filtered_bboxes = Filtering.filter_bboxes_by_semantic_lidar(
            bbox_list, semantic_detections, distance_threshold=2.0
        )
        
        return filtered_bboxes
    
    ### DISTANCE FILTERING & FOV FILTERING
    @staticmethod
    def distance_filter(bbox_list, lidar_transform, max_range=150.0, verbose=True):
        if not bbox_list or lidar_transform is None:
            return bbox_list
        try:
            max_range = float(max_range)
        except (ValueError, TypeError):
            max_range = 150.0
        
        lidar_location = lidar_transform.location
        lidar_pos = np.array([lidar_location.x, lidar_location.y, lidar_location.z])
        
        filtered_bboxes = []
        filtered_count = 0
        
        for i, bbox in enumerate(bbox_list):
            try:
                center_3d = bbox.get('center_3d')
                if center_3d is None:
                    continue
                
                # center_3d 
                if isinstance(center_3d, dict):
                    if 'x' in center_3d and 'y' in center_3d and 'z' in center_3d:
                        bbox_pos = np.array([center_3d['x'], center_3d['y'], center_3d['z']])
                    else:
                        continue
                elif isinstance(center_3d, (list, tuple)) and len(center_3d) >= 3:
                    bbox_pos = np.array([float(center_3d[0]), float(center_3d[1]), float(center_3d[2])])
                elif isinstance(center_3d, np.ndarray) and len(center_3d) >= 3:
                    bbox_pos = center_3d[:3].astype(float)
                else:
                    continue
                
                distance = float(np.linalg.norm(bbox_pos - lidar_pos))
                
                if distance <= max_range:
                    filtered_bboxes.append(bbox)
                else:
                    filtered_count += 1
                        
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
        
        return filtered_bboxes

    @staticmethod
    def check_points_in_screen(vertices_2d, width, height, min_visible_points=1):
        """ 흠"""
        if not isinstance(vertices_2d, np.ndarray):
            vertices_2d = np.array(vertices_2d)
        
        border_x = width * 0.1
        border_y = height * 0.1
        
        in_screen_points = np.sum(
            (vertices_2d[:, 0] >= -border_x) & 
            (vertices_2d[:, 0] < width + border_x) & 
            (vertices_2d[:, 1] >= -border_y) & 
            (vertices_2d[:, 1] < height + border_y)
        )
        
        return in_screen_points >= min_visible_points
    
    # 
    @staticmethod
    def filter_by_camera_view(object_location, camera_sensor):
        """camera fov filtering """
        camera_transform = camera_sensor.get_transform()
        camera_location = camera_transform.location
        camera_forward = camera_transform.get_forward_vector()
        
        dist = object_location.distance(camera_location)
        if dist > 200:
            return False
            
        object_vector = object_location - camera_location
        object_vector_normalized = object_vector / object_vector.length()
        forward_dot = camera_forward.dot(object_vector_normalized)
        if forward_dot <= -0.5:
            return False
            
        if hasattr(camera_sensor, 'fov'):
            h_fov = float(camera_sensor.fov)
        else:
            h_fov = float(camera_sensor.attributes.get('fov', 90.0))
        
        h_fov = h_fov * 1.4
    
        # vfov
        width = int(camera_sensor.attributes.get('image_size_x', 1920))
        height = int(camera_sensor.attributes.get('image_size_y', 1080))
        aspect_ratio = width / height
        v_fov = h_fov / aspect_ratio
        v_fov = v_fov * 1.4
        
        h_fov_rad = math.radians(h_fov / 2)
        v_fov_rad = math.radians(v_fov / 2)
        
        camera_right = camera_transform.get_right_vector()
        camera_up = camera_transform.get_up_vector()
        right_dot = camera_right.dot(object_vector_normalized)
        up_dot = camera_up.dot(object_vector_normalized)
        
        try:
            if abs(forward_dot) < 0.01:
                horizontal_angle = 0 if abs(right_dot) < 0.01 else math.pi/2
                vertical_angle = 0 if abs(up_dot) < 0.01 else math.pi/2
            else:
                horizontal_angle = math.asin(min(1.0, max(-1.0, right_dot / forward_dot)))
                vertical_angle = math.asin(min(1.0, max(-1.0, up_dot / forward_dot)))
        except Exception as e:
            return True
        
        if abs(horizontal_angle) > h_fov_rad * 1.2 or abs(vertical_angle) > v_fov_rad * 1.2:
            return False
            
        return True

    ### Box Volume Filtering
    @staticmethod
    def calculate_obb_volume(bbox_data):
        """OBB(Oriented Bounding Box) volume"""
        try:
            size_3d = bbox_data.get('size_3d', [0, 0, 0])
            volume = size_3d[0] * size_3d[1] * size_3d[2]
            return volume
        except:
            return 0.0
        
    @staticmethod
    def filter_static_vehicles_obb(static_bboxes, overlap_threshold=0.7):
        if not static_bboxes: 
            return []
            
        sorted_bboxes = sorted(static_bboxes, 
                              key=lambda x: Filtering.calculate_obb_volume(x), 
                              reverse=True)
        
        filtered_bboxes = []
        used_indices = set()
        
        for i, bbox1 in enumerate(sorted_bboxes):
            if i in used_indices:
                continue
                
            filtered_bboxes.append(bbox1)
            used_indices.add(i)
            
            center1 = np.array(bbox1['center_3d'])
            size1 = np.array(bbox1['size_3d'])
            
            for j, bbox2 in enumerate(sorted_bboxes):
                if j in used_indices or j <= i:
                    continue
                
                center2 = np.array(bbox2['center_3d'])
                size2 = np.array(bbox2['size_3d'])
                
                distance = np.linalg.norm(center1 - center2)
                max_allowed_distance = (np.linalg.norm(size1) + np.linalg.norm(size2)) / 4.0
                
                if distance < max_allowed_distance:
                    used_indices.add(j)

        return filtered_bboxes

    
class Transform:
    ### DEPTH TRANSFORM
    @staticmethod
    def world_to_camera_with_depth(world_point, camera_sensor):
        """
        World 좌표를 Camera 좌표계로 변환 +  depth 정보
        
        Args:
            world_point: carla.Location 또는 [x, y, z] 좌표
            camera_sensor: CARLA camera sensor
            
        Returns:
            dict: {
                'camera_coords': [x, y, z],  # Camera 좌표계에서의 3D 좌표
                'image_coords': [u, v],      # 이미지 평면에서의 2D 좌표  
                'depth': float,              # Camera에서의 거리
                'valid': bool                # 투영 성공 여부
            }
        """
        try:
            # World point 정규화
            if hasattr(world_point, 'x'):  # carla.Location
                world_loc = world_point
            else:  # list 또는 tuple
                world_loc = carla.Location(x=world_point[0], y=world_point[1], z=world_point[2])
            
            # Camera 변환 행렬
            camera_transform = camera_sensor.get_transform()
            world_2_camera = np.array(camera_transform.get_inverse_matrix())
            
            # World → Camera 좌표계 변환
            world_homo = np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])
            camera_homo = world_2_camera @ world_homo
            camera_coords = camera_homo[:3]
            
            # CARLA UE4 → Standard Camera 좌표계 변환
            # CARLA: X=front, Y=right, Z=up → Camera: X=right, Y=down, Z=forward
            standard_camera_coords = np.array([
                camera_coords[1],    # X = Y (right)
                -camera_coords[2],   # Y = -Z (down)  
                camera_coords[0]     # Z = X (forward/depth)
            ])
            
            # Depth (forward distance)
            depth = standard_camera_coords[2]
            
            # Camera가 객체 뒤에 있으면 invalid
            if depth <= 0:
                return {
                    'camera_coords': standard_camera_coords.tolist(),
                    'image_coords': None,
                    'depth': depth,
                    'valid': False
                }
            
            # Image plane projection
            K = np.array(camera_sensor.calibration)
            image_coords_homo = K @ standard_camera_coords
            
            if image_coords_homo[2] != 0:
                u = image_coords_homo[0] / image_coords_homo[2]
                v = image_coords_homo[1] / image_coords_homo[2]
                image_coords = [u, v]
            else:
                image_coords = None
            
            return {
                'camera_coords': standard_camera_coords.tolist(),
                'image_coords': image_coords,
                'depth': depth,
                'valid': image_coords is not None and depth > 0
            }
            
        except Exception as e:
            print(f"Error in world_to_camera_with_depth: {e}")
            return {
                'camera_coords': None,
                'image_coords': None,
                'depth': None,
                'valid': False
            }
    
    @staticmethod
    def validate_depth_consistency(world_point, camera_sensor, depth_array, tolerance=1.0):
        """
        World 좌표의 객체가 depth image와 일치하는지 검증 
            tolerance: 허용 거리 오차 (미터)
        """
        try:
            # 1. World → Camera Proj
            projection_result = Transform.world_to_camera_with_depth(world_point, camera_sensor)
            
            if not projection_result['valid']:
                return {
                    'valid': False,
                    'actual_depth': None,
                    'image_depth': None,
                    'difference': None,
                    'pixel_coords': None,
                    'reason': 'projection_failed'
                }
            
            image_coords = projection_result['image_coords']
            actual_depth = projection_result['depth']
            
            
            u, v = int(image_coords[0]), int(image_coords[1])
            height, width = depth_array.shape
            
            if u < 0 or u >= width or v < 0 or v >= height:
                return {
                    'valid': False,
                    'actual_depth': actual_depth,
                    'image_depth': None,
                    'difference': None,
                    'pixel_coords': [u, v],
                    'reason': 'out_of_bounds'
                }
            
            # 3. Depth Value (여기서 meter단위로 이미변환댐)
            image_depth = depth_array[v, u]
            
            # 4. 거리 비교 
            depth_difference = abs(actual_depth - image_depth)
            is_valid = depth_difference <= tolerance
            
            return {
                'valid': is_valid,
                'actual_depth': actual_depth,
                'image_depth': image_depth,
                'difference': depth_difference,
                'pixel_coords': [u, v],
                'reason': 'validated' if is_valid else f'depth_mismatch_{depth_difference:.2f}m'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'actual_depth': None,
                'image_depth': None,
                'difference': None,
                'pixel_coords': None,
                'reason': f'error_{str(e)}'
            }
    
    @staticmethod
    def get_bbox_camera_validation_score(bbox, camera_sensors, depth_data_dict, min_validation_ratio=0.5):
        """
        bbox 검증 (7대 카메라)
        """
        if not camera_sensors or not depth_data_dict:
            print(f"No camera/depth data available for validation")
            return {
                'score': 1.0,  # depth 데이터 없으면 Pass
                'validated_cameras': 0,
                'total_cameras': 0,
                'valid': True,
                'details': []
            }
        
        # BBox 중심점
        world_center = carla.Location(
            x=bbox['center_3d'][0], 
            y=bbox['center_3d'][1], 
            z=bbox['center_3d'][2]
        )
        validated_cameras = 0
        total_cameras = 0
        details = []
        
        for camera_id, camera_sensor in camera_sensors.items():
            depth_key = f"depth_{camera_id}"  # camera0 → depth_camera0
            if depth_key not in depth_data_dict:
                continue
                
            depth_array = depth_data_dict[depth_key]
            if depth_array is None:
                continue
            total_cameras += 1
            
            validation_result = Transform.validate_depth_consistency(
                world_center, camera_sensor, depth_array, tolerance=2.0
            )
            
            details.append({
                'camera_id': camera_id,
                'depth_key': depth_key,
                'validation_result': validation_result
            })
            
            if validation_result['valid']:
                validated_cameras += 1
            else:
                continue
        if total_cameras == 0:     
            score = 1.0
            is_valid = True
        else:
            score = validated_cameras / total_cameras
            is_valid = score >= min_validation_ratio
        return {
            'score': score,
            'validated_cameras': validated_cameras,
            'total_cameras': total_cameras,
            'valid': is_valid,
            'details': details
        }

    @staticmethod
    def get_carla_transform_matrix(transform):
        """
        CARLA Transform을 4x4 matrix로 변환 
        회전 순서: Yaw(Z) → Pitch(Y) → Roll(X)
        """
        rotation = transform.rotation
        location = transform.location
        
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        
        matrix = np.eye(4)
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        
        return matrix

    @staticmethod
    def rotation_matrix_to_euler_zyx(R):
        sin_pitch = R[2, 0]
        if abs(sin_pitch) >= 0.99999:
            yaw = np.arctan2(-R[0, 1], R[1, 1])
            pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
            roll = 0.0
        else:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
            roll = np.arctan2(-R[2, 1], R[2, 2])
        
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    

    @staticmethod
    def lidar_to_camera_ros(lidar_sensor, camera_sensor):
        """
        LiDAR → Camera 변환 (ROS Camera Optical 좌표계)
        
        CARLA: X=front, Y=right, Z=up (왼손좌표계)
        ROS Camera Optical: X=right, Y=down, Z=forward (오른손좌표계)
        """
        lidar_to_world = Transform.get_carla_transform_matrix(lidar_sensor.get_transform())
        world_to_camera = np.linalg.inv(Transform.get_carla_transform_matrix(camera_sensor.get_transform()))
        
        # LiDAR → Camera (CARLA)
        lidar_to_camera_carla = world_to_camera @ lidar_to_world
        
        # CARLA → ROS Camera Optical 
        # CARLA (X=front, Y=right, Z=up) → ROS Camera Optical (X=right, Y=down, Z=forward)
        T_lidar_carla_to_ros = np.diag([1, -1, 1, 1])  # Y axis flip
        T_carla_to_ros_cam = np.array([
            [ 0, 1,  0,  0],  # X_ros = Y_carla (right)
            [ 0,  0, -1,  0],  # Y_ros = -Z_carla (down)  
            [ 1,  0,  0,  0],  # Z_ros = X_carla (forward)
            [ 0,  0,  0,  1]
        ])
        
        # 최종 LiDAR(CARLA) → Camera(CARLA) → Camera(ROS)
        lidar_to_camera_ros = T_carla_to_ros_cam @ lidar_to_camera_carla @ T_lidar_carla_to_ros
        
        return lidar_to_camera_ros

    @staticmethod
    def lidar2cam_optical(lidar: carla.Actor, cam: carla.Actor):
        
        E_matrix = Transform.lidar_to_camera_ros(lidar, cam)

        xyz = E_matrix[:3, 3]
        quat = transforms3d.quaternions.mat2quat(E_matrix[:3, :3])
        quat = [quat[1], quat[2], quat[3], quat[0]]  # (x,y,z,w)
        
        return E_matrix, xyz, quat
    
    @staticmethod 
    def world_to_lidar(bbox_vertices_world, lidar_transform):
        """World → LiDAR coord transform """
        if lidar_transform is None:
            return np.array([[v.x, v.y, v.z] for v in bbox_vertices_world])
            
        lidar_inv_matrix = np.linalg.inv(Transform.get_carla_transform_matrix(lidar_transform))
        
        vertices_lidar = []
        for vertex in bbox_vertices_world: 
            if hasattr(vertex, 'x'):
                world_homo = np.array([vertex.x, vertex.y, vertex.z, 1.0])
            else:
                world_homo = np.array([vertex[0], vertex[1], vertex[2], 1.0])
                
            lidar_homo = lidar_inv_matrix @ world_homo
            vertices_lidar.append(lidar_homo[:3])
        
        vertices_lidar = np.array(vertices_lidar)
        # CARLA → ROS : Y축 반전
        vertices_lidar[:, 1] = -vertices_lidar[:, 1]  
        
        return vertices_lidar
    
    # INTRINSIC
    @staticmethod
    def build_projection_matrix(w, h, fov, is_behind_camera=False):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)

        if is_behind_camera:
            K[0, 0] = K[1, 1] = -focal
        else:
            K[0, 0] = K[1, 1] = focal

        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    @staticmethod
    def get_image_point(loc, K, w2c):
        point = np.array([loc.x, loc.y, loc.z, 1])
        point_camera = np.dot(w2c, point)
        
        # CARLA UE4 좌표계 변환: (x, y, z) -> (y, -z, x) : 이게  ROS 카메라 보정임 ㅋㅋㅋ 
        point_camera = np.array([
            point_camera[1],    # y
            -point_camera[2],   # -z
            point_camera[0]     # x
        ])
        
        point_img = np.dot(K, point_camera)
        
        if point_img[2] != 0:
            point_img[0] /= point_img[2]
            point_img[1] /= point_img[2]

        return point_img[0:2]
    