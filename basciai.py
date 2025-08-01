import numpy as np
import carla
import json
import uuid
import os
from datetime import datetime
from ontology import OntologyMapper
from typing import Optional, Dict, List, Any

class BasicAIAnnotator:    
    def __init__(self, ontology_file_path: Optional[str] = None):
        self.ontology_mapper = None
        self.use_ontology = False
        
        if ontology_file_path:
            self.initialize_ontology(ontology_file_path)
        else:
            self._prompt_for_ontology()
    
    def _prompt_for_ontology(self):
        print("=" * 80)
        print("BasicAI Ontology Configuration")
        print("=" * 80)
        print("Please provide the path to your ontology.json file.")
        print("You can:")
        print("1. Enter the full path to your ontology.json file")
        print("2. Press Enter to continue without ontology (basic annotations only)")
        print("=" * 80)
        
        while True:
            try:
                user_input = input("Ontology file path (or press Enter to skip): ").strip()
                
                if not user_input:
                    print("Continuing without ontology. Basic annotations will be generated.")
                    self.use_ontology = False
                    return
                
                if os.path.exists(user_input):
                    self.initialize_ontology(user_input)
                    return
                else:
                    print(f"File not found: {user_input}")
                    print("Please check the path and try again, or press Enter to skip.")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled. Continuing without ontology.")
                self.use_ontology = False
                return
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again or press Enter to skip.")
    
    def initialize_ontology(self, ontology_file_path: str):
        try:
            self.ontology_mapper = OntologyMapper(ontology_file_path)
            self.use_ontology = True
            print(f"BasicAIAnnotator initialized with ontology: {ontology_file_path}")
        except Exception as e:
            print(f"Failed to initialize ontology: {e}")
            print("Would you like to continue without ontology? (y/n): ", end="")
            try:
                response = input().strip().lower()
                if response in ['y', 'yes', '']:
                    print("Continuing without ontology. Basic annotations will be generated.")
                    self.use_ontology = False
                else:
                    raise RuntimeError("Ontology initialization failed and user chose not to continue")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                raise RuntimeError("Ontology initialization cancelled by user")
    
    def get_class_info(self, carla_class_id: int) -> Dict:
        if self.use_ontology and self.ontology_mapper:
            return self.ontology_mapper.get_class_info(carla_class_id)
        else:
            basic_class_mapping = {
                0: {"classId": 1, "className": "Vehicle"},
                1: {"classId": 2, "className": "Bicycle"},
                2: {"classId": 3, "className": "Pedestrian"}
            }
            return basic_class_mapping.get(carla_class_id, basic_class_mapping[0])
    
    def get_class_values(self, carla_class_id: int, semantic_tag: Optional[int] = None) -> Optional[List[Dict]]:
        if self.use_ontology and self.ontology_mapper:
            return self.ontology_mapper.get_class_values(carla_class_id, semantic_tag)
        else:
            basic_values = {
                0: [{"id": 1, "name": "Vehicle", "value": "car", "type": "option"}],
                1: [{"id": 2, "name": "Vehicle", "value": "bicycle", "type": "option"}],
                2: [{"id": 3, "name": "Human", "value": "adult", "type": "option"}]
            }
            return basic_values.get(carla_class_id, basic_values[0])
    
    def get_class_version(self, carla_class_id: int, is_static: bool = False) -> int:
        if self.use_ontology and self.ontology_mapper:
            return self.ontology_mapper.get_class_version(carla_class_id, is_static)
        else:
            return 1
    
    def ensure_json_serializable(self, value):
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, list):
            return [self.ensure_json_serializable(item) for item in value]
        elif isinstance(value, dict):
            return {k: self.ensure_json_serializable(v) for k, v in value.items()}
        else:
            return value
    
    def process_all_cameras_for_basicai(self, camera_configs, lidar_sensor, frame_id):
        from bbox import Transform  
        
        camera_config_list = []
        
        for cam_config in camera_configs:
            camera_id = cam_config["sensor_config"]["id"]
        
            if camera_id == "camera7":
                continue
                
            sensor_config = cam_config["sensor_config"]
            sensor_spawned = cam_config["sensor_spawned"]
            
            try:
                # Intrinsic
                width = int(sensor_config.get("image_size_x", 1920))
                height = int(sensor_config.get("image_size_y", 1080))
                fov = float(sensor_config.get("fov", 90.0))
                
                if hasattr(sensor_spawned, 'calibration') and sensor_spawned.calibration is not None:
                    K = sensor_spawned.calibration
                    intrinsics = {
                        "fx": float(K[0, 0]),
                        "fy": float(K[1, 1]),
                        "cx": float(K[0, 2]),
                        "cy": float(K[1, 2]),
                        "width": width,
                        "height": height
                    }
                else:
                    # Fallback
                    camera_yaw = sensor_config.get("spawn_point", {}).get("yaw", 0.0)
                    is_behind_camera = (camera_yaw > 90.0) or (camera_yaw < -90.0)
                    
                    K = Transform.build_projection_matrix(width, height, fov, is_behind_camera)
                    
                    intrinsics = {
                        "fx": float(K[0, 0]),
                        "fy": float(K[1, 1]),
                        "cx": float(K[0, 2]),
                        "cy": float(K[1, 2]),
                        "width": width,
                        "height": height
                    }
                
                # Lidar 기준 extrinsic
                extrinsic_matrix = Transform.lidar_to_camera_ros(lidar_sensor, sensor_spawned)

                # Column-major
                extrinsic_array = extrinsic_matrix.flatten('F').tolist()
                
                camera_entry = {
                    "camera_internal": {
                        "fx": intrinsics["fx"],
                        "fy": intrinsics["fy"], 
                        "cx": intrinsics["cx"],
                        "cy": intrinsics["cy"]
                    },
                    "width": intrinsics["width"],
                    "height": intrinsics["height"],
                    "camera_external": extrinsic_array,
                    "rowMajor": False  # Column-major
                }
                
                camera_config_list.append(camera_entry)
                    
            except Exception as e:
                print(f"Error generating camera config for {camera_id}: {e}")
                import traceback
                traceback.print_exc()
        
        
        return camera_config_list
    
    def create_annotation(self, bbox_list, save_path, frame_id, lidar_transform=None, 
                         track_uuid_manager=None, camera_configs=None, camera_config_save_path=None, 
                         source_id=None, source_name=None):
        from bbox import Transform
        
        if not bbox_list:
            empty_annotation_data = [{
                "version": "1.0",
                "dataId": frame_id,
                "sourceId": source_id if source_id is not None else 1470017,
                "sourceType": "EXTERNAL_GROUND_TRUTH", 
                "sourceName": source_name if source_name is not None else f"frame_{frame_id:08d}",
                "validity": "UNKNOWN",
                "classifications": [],
                "instances": [],
                "segments": [],
                "entities": None,
                "relations": None
            }]
            
            bbox_output_file = os.path.join(save_path, f'{frame_id:08d}.json')
            with open(bbox_output_file, 'w') as f:
                json.dump(empty_annotation_data, f, indent=2)
            print(f"Frame {frame_id}: Saved empty File 1 annotation to {bbox_output_file}")
            return
            
        instances = []
        
        if lidar_transform:
            print(f"LiDAR transform - Location: {lidar_transform.location}, Rotation: {lidar_transform.rotation}")
        
        for bbox in bbox_list:
            unique_id = str(uuid.uuid4())
            
            # ===== CARLA World → CARLA LiDAR → ROS LiDAR =====
            if lidar_transform is not None:
                # 1) Center (CARLA World → CARLA LiDAR)
                world_center = carla.Location(
                    x=bbox['center_3d'][0], 
                    y=bbox['center_3d'][1], 
                    z=bbox['center_3d'][2]
                )
                
                lidar_inv_matrix = np.linalg.inv(Transform.get_carla_transform_matrix(lidar_transform))
                world_homo = np.array([world_center.x, world_center.y, world_center.z, 1.0])
                carla_lidar_homo = lidar_inv_matrix @ world_homo
                
                # 2) CARLA LiDAR → ROS LiDAR 
                center_lidar = [
                    carla_lidar_homo[0],     # X (forward) 동일
                    -carla_lidar_homo[1],    # Y (right → left, 부호 반전)
                    carla_lidar_homo[2]      # Z (up) 동일
                ]
                center_3d = center_lidar
                
                # Vertices 
                vertices_corrected = Transform.world_to_lidar(
                    bbox['vertices_world'], lidar_transform
                )
                vertices_lidar = vertices_corrected.flatten().tolist()
                
                transform = bbox.get('transform')
                
                if transform:
                    # Dynamic Object 
                    world_rotation = transform.rotation
                    lidar_rotation = lidar_transform.rotation 
                    world_R = Transform.get_carla_transform_matrix(transform)[:3, :3]
                    lidar_R = Transform.get_carla_transform_matrix(lidar_transform)[:3, :3]
                    
                    # R_relative = R_lidar^(-1) * R_world
                    relative_R = np.linalg.inv(lidar_R) @ world_R
                    
                    #  EULER : (ZYX)
                    relative_roll, relative_pitch, relative_yaw = Transform.rotation_matrix_to_euler_zyx(relative_R)
                    
                else:
                    # Static Object
                    world_yaw = bbox.get('yaw', 0.0)
                    lidar_yaw = lidar_transform.rotation.yaw
                    relative_yaw = world_yaw - lidar_yaw
                    relative_pitch = 0.0
                    relative_roll = 0.0
                
                # CARLA LiDAR → ROS LiDAR 
                ros_roll = relative_roll      
                ros_pitch = -relative_pitch   
                ros_yaw = -relative_yaw       
                
                # Degrees → Radians
                roll_rad = np.deg2rad(ros_roll)
                pitch_rad = np.deg2rad(ros_pitch)
                yaw_rad = np.deg2rad(ros_yaw)
                    
            else:
                print(f"ERROR: No LiDAR transform provided for frame {frame_id}")
                return
            
            # Class mapping by Ontology
            class_id = bbox['class_id']
            class_info = self.get_class_info(class_id)
            semantic_tag = bbox.get('semantic_tag')
            class_values = self.get_class_values(class_id, semantic_tag)
            
            # Track
            object_track_id = bbox.get('track_id', bbox['actor_id'])
            track_name = str(object_track_id)
            track_id = track_uuid_manager.get_track_uuid(object_track_id) if track_uuid_manager else str(uuid.uuid4())
        
            # Size 
            size_3d = bbox['size_3d']
            
            class_version = self.get_class_version(class_id, bbox.get('is_static', False))
            
            # ROS LiDAR coord
            instance = {
                "trackId": track_id,
                "trackName": track_name,
                "groups": [],  
                "contour": {
                    "center3D": {
                        "x": float(center_3d[0]),
                        "y": float(center_3d[1]),
                        "z": float(center_3d[2])
                    },
                    "pointN": int(len(vertices_lidar) // 3),  # int() 명시적 변환
                    "points": [],  
                    "rotation3D": {
                        "x": float(roll_rad),  
                        "y": float(pitch_rad),  
                        "z": float(yaw_rad)    
                    },
                    "sensorDistance": float(np.linalg.norm(center_3d)),  # 파일 1에 있음
                    "size3D": {
                        "x": float(size_3d[0]),
                        "y": float(size_3d[1]),
                        "z": float(size_3d[2])
                    }
                },
                "modelConfidence": None,  
                "modelClass": "Car" if class_id == 0 else ("Pedestrian" if class_id == 2 else "Car"),  # 파일 1 스타일
                "classVersion": int(class_version),  
                "isValid": None,
                "note": None,
                "start": None,
                "end": None,
                "deviceName": "lidar_point_cloud_0",  
                "deviceFrame": None,
                "bevFrameName": None,
                "index": None,
                "role": None,
                "content": None,
                "id": unique_id,
                "type": "3D_BOX",
                "classId": int(class_info["classId"]),    
                "className": class_info["className"],     
                "classNumber": None,
                "classValues": self.ensure_json_serializable(class_values), 
                "createdAt": 1750819713000,  
                "createdBy": 870005          
            }
            
            instance = self.ensure_json_serializable(instance)
            instances.append(instance)
        
        annotation_data = [{
            "version": "1.0",
            "dataId": frame_id,
            "sourceId": source_id if source_id is not None else 1470017,
            "sourceType": "EXTERNAL_GROUND_TRUTH",
            "sourceName": source_name if source_name is not None else f"frame_{frame_id:08d}",
            "validity": "UNKNOWN", 
            "classifications": [], 
            "instances": instances,
            "segments": [],  
            "entities": None,
            "relations": None
        }]
        
        bbox_output_file = os.path.join(save_path, f'{frame_id:08d}.json')
        with open(bbox_output_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        print(f"Frame {frame_id}: Saved {len(instances)} bbox instances to {bbox_output_file} in File 1 format")

_global_annotator = None

def get_basicai_annotator(ontology_file_path: Optional[str] = None) -> BasicAIAnnotator:
    global _global_annotator
    
    if ontology_file_path is None:
        import os
        ontology_file_path = os.environ.get('ONTOLOGY_FILE_PATH', '')
        
        if not ontology_file_path:
            ontology_file_path = "./ontology_config/ontology.json"
    
    if _global_annotator is None:
        _global_annotator = BasicAIAnnotator(ontology_file_path)
    elif not _global_annotator.use_ontology and ontology_file_path:
        _global_annotator.initialize_ontology(ontology_file_path)
    
    return _global_annotator