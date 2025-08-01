import queue
import time
import sys
import cv2
import numpy as np
import carla
import os
import random
import open3d as o3d
from lidar import VID_RANGE, VIRIDIS

from PIL import Image
from constants import LABEL_COLORS
from bbox import BoundingBox, ObjectTracker, Transform, TrackUUIDManager, Filtering
from visualization import SensorProcessor
from lidar import MultiOT128Processor  
from nuscenes import NuScenesExporter
from basciai import get_basicai_annotator



import json
import re 

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context.
    """
    def __init__(self, world, data_vehicle, vehicle, stop_time, client=None, save_root="./nuscenes", scene_name="carla_scene", **kwargs):
        self.world = world
        self.vehicle = vehicle
        self.client=client
        sensor_list, sensor_dict = self.setup_vehicle_sensors(self.world, data_vehicle, self.vehicle)
        self.sensors = sensor_list
        self.data_dict = sensor_dict
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 30)
        settings = world.get_settings()
        settings.fixed_delta_seconds = self.delta_seconds 
        world.apply_settings(settings)
        print('delta sec : ', self.delta_seconds)
        self._queues = []
        self._settings = None
        self.ego_vehicle_id = vehicle.id
        self.vehicle_ids = set()
        self.vehicle = vehicle
        self.last_location = None
        self.last_move_time = None
        self.stop_time = stop_time
        self.frame_counter = 0 
        self.track_uuid_manager = TrackUUIDManager()
        # Basic AI format id 
        self.basic_ai_source_id = random.randint(100000, 9999999)  # 숫자만
        self.basic_ai_source_name = str(random.randint(100000000, 999999999))  # 숫자 문자열

        self.save_root = save_root 

        # Traffic Manager 
        print("Initializing Traffic Manager...")
        try:
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_synchronous_mode(True)
            self.vehicle.set_autopilot(True)
            self.traffic_manager.set_desired_speed(self.vehicle, 40)
            self.traffic_manager.set_global_distance_to_leading_vehicle(4.0)
            print("Traffic Manager initialized successfully")
        # 이거 지울까말까 고민 
        except Exception as e:
            print(f"Traffic Manager initialization failed: {e}")
            print("TRAFFIC_MANAGER_FAILED")  # VEHICLE_STOPPED
            sys.stdout.flush()
            self.traffic_manager_failed = True
            self.traffic_manager = None
        else:
            self.traffic_manager_failed = False

        # # Traffic Manager 
        # self.traffic_manager = self.client.get_trafficmanager()
        # self.traffic_manager.set_synchronous_mode(True)
        # self.vehicle.set_autopilot(True)
        # self.traffic_manager.set_desired_speed(self.vehicle, 40)
        # self.traffic_manager.set_global_distance_to_leading_vehicle(4.0)
            
        self.collect_static_vehicles()
        
        self.object_tracker = ObjectTracker()
        
        self.basic_ai_save_path = os.path.join(save_root, 'basic_ai')
        os.makedirs(self.basic_ai_save_path, exist_ok=True)
        self.camera_config_save_path = os.path.join(save_root, 'camera_config')
        os.makedirs(self.camera_config_save_path, exist_ok=True)

        self.lidar_sensors = {}
        for sensor_dict in self.data_dict:
            if sensor_dict["type"] == "sensor.lidar.ray_cast":
                self.lidar_sensors[sensor_dict["id"]] = sensor_dict["sensor_spawned"]
    
        self.camera_configs = []
        for sensor_dict in self.data_dict:
            if sensor_dict["type"] == "sensor.camera.rgb":
                self.camera_configs.append({
                    "sensor_config": sensor_dict, 
                    "sensor_spawned": sensor_dict["sensor_spawned"]  
                })

        # Semantic LiDAR
        self.semantic_lidar_sensor = None
        self.current_semantic_data = None
        for sensor_dict in self.data_dict:
            if sensor_dict["type"] == "sensor.lidar.ray_cast_semantic":
                self.semantic_lidar_sensor = sensor_dict["sensor_spawned"]
                print(f"Found Semantic LiDAR sensor: {sensor_dict['id']}")
                break

        print(f"Found {len(self.lidar_sensors)} LiDAR sensors: {list(self.lidar_sensors.keys())}")
        
        self.primary_lidar = list(self.lidar_sensors.values())[0] if self.lidar_sensors else None
        
        # NuScenes exporter --------------------------------------------------
        start_us = int(time.time()*1e6)
        self.nusc = NuScenesExporter(save_root, scene_name)
        self.scene_token = self.nusc.register_scene(start_us)

        # (sensor_id → (sensor_token, calib_token))
        self.sensor_tokens = {}
    
        # NuScenes용
        self._register_sensors_to_nuscenes()

    def setup_vehicle_sensors(self, world, sensors, parent_actor=None):
        """Setup sensors on the vehicle"""
        def create_spawn_point(x, y, z, roll, pitch, yaw):
            return carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(roll=roll, pitch=pitch, yaw=yaw))

        vehicle_sensors = []
        sensor_dict = []

        for idx, sensor_spec in enumerate(sensors):
            try:
                sensor_type = str(sensor_spec["type"])
                sensor_id = str(sensor_spec["id"])

                # Get blueprint
                bp = world.get_blueprint_library().find(sensor_type)
                if bp is None:
                    print(f"Could not find blueprint for sensor type: {sensor_type}")
                    continue

                # Setup spawn point
                spawn_point = sensor_spec.get("spawn_point", None)
                if spawn_point is None:
                    sensor_transform = create_spawn_point(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    sensor_transform = create_spawn_point(
                        spawn_point.get("x", 0.0),
                        spawn_point.get("y", 0.0),
                        spawn_point.get("z", 0.0),
                        spawn_point.get("roll", 0.0),
                        spawn_point.get("pitch", 0.0),
                        spawn_point.get("yaw", 0.0))

                if sensor_type.startswith("sensor.camera"):
                    if "image_size_x" in sensor_spec:
                        bp.set_attribute("image_size_x", str(sensor_spec["image_size_x"]))
                    if "image_size_y" in sensor_spec:
                        bp.set_attribute("image_size_y", str(sensor_spec["image_size_y"]))
                    if "fov" in sensor_spec:
                        bp.set_attribute("fov", str(sensor_spec["fov"]))

                elif sensor_type == "sensor.lidar.ray_cast":
                    lidar_attrs = ["range", "channels", "points_per_second",
                                "rotation_frequency", "upper_fov", "lower_fov",
                                "atmosphere_attenuation_rate",
                                "dropoff_general_rate",
                                "dropoff_intensity_limit",
                                "dropoff_zero_intensity",
                                "noise_stddev"]
                    for attr in lidar_attrs:
                        if attr in sensor_spec:
                            bp.set_attribute(attr, str(sensor_spec[attr]))

                elif sensor_type == "sensor.lidar.ray_cast_semantic":  # 추가!
                    semantic_attrs = ["range", "channels", "points_per_second",
                                    "rotation_frequency", "upper_fov", "lower_fov"]
                    for attr in semantic_attrs:
                        if attr in sensor_spec:
                            bp.set_attribute(attr, str(sensor_spec[attr]))
      
                # elif sensor_type == "sensor.other.radar":
                #     radar_attrs = ["horizontal_fov", "vertical_fov", "range", "points_per_second"]
                #     for attr in radar_attrs:
                #         if attr in sensor_spec:
                #             bp.set_attribute(attr, str(sensor_spec[attr]))

                # elif sensor_type == "sensor.other.gnss":
                #     gnss_attrs = ["noise_alt_stddev", "noise_lat_stddev", "noise_lon_stddev",
                #                 "noise_alt_bias", "noise_lat_bias", "noise_lon_bias"]
                #     for attr in gnss_attrs:
                #         if attr in sensor_spec:
                #             bp.set_attribute(attr, str(sensor_spec[attr]))

                # Spawn sensor
                sensor_spawned = world.spawn_actor(bp, sensor_transform, attach_to=parent_actor)
                
                if sensor_type == "sensor.camera.rgb":
                    image_w = int(sensor_spec.get("image_size_x", 1920))
                    image_h = int(sensor_spec.get("image_size_y", 1080))
                    fov = float(sensor_spec.get("fov", 90.0))
                    
                    K = Transform.build_projection_matrix(image_w, image_h, fov)
                    sensor_spawned.calibration = K

                vehicle_sensors.append(sensor_spawned)
                sensors[idx]["sensor_spawned"] = sensor_spawned
                sensor_dict.append(sensors[idx])
                print(f"Successfully spawned {sensor_type} with ID: {sensor_id}")

            except Exception as e:
                print(f"Error spawning sensor {sensor_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return vehicle_sensors, sensor_dict

    def _register_sensors_to_nuscenes(self):
        for sensor_dict in self.data_dict:
            if sensor_dict["type"].startswith("sensor.camera") or sensor_dict["type"].startswith("sensor.lidar"):
                # Sensor blueprint 
                sensor_bp = self.world.get_blueprint_library().find(sensor_dict["type"])
                
                # sensor transform
                spawn_point = sensor_dict.get("spawn_point", {})
                sensor_transform = carla.Transform(
                    carla.Location(x=spawn_point.get("x", 0.0), y=spawn_point.get("y", 0.0), z=spawn_point.get("z", 0.0)),
                    carla.Rotation(roll=spawn_point.get("roll", 0.0), pitch=spawn_point.get("pitch", 0.0), yaw=spawn_point.get("yaw", 0.0))
                )
                
                K = getattr(sensor_dict["sensor_spawned"], 'calibration', np.eye(3))
                
                self.nusc.add_sensor(
                    sensor_dict["id"], 
                    sensor_bp, 
                    sensor_transform, 
                    K
                )

    ## Traffic Manager + Vehicle stopped
    def check_traffic_manager_status(self):
        if hasattr(self, 'traffic_manager_failed') and self.traffic_manager_failed:
            print("TRAFFIC_MANAGER_FAILED")
            sys.stdout.flush()
            return False
        return True

    def check_vehicle_movement(self):
        if not self.check_traffic_manager_status():
            return False
            
        current_location = self.vehicle.get_location()
        current_time = time.time()

        if not self.last_location:
            self.last_location = current_location
            self.last_move_time = current_time
            return True
        
        # vehicle stop
        if current_location.distance(self.last_location) < 0.1:    # 0.1m 이하 이동 -> 정지 처리 
            if current_time - self.last_move_time > self.stop_time:     # n초 이상 정지시
                print("VEHICLE_STOPPED")
                sys.stdout.flush()
                return False
        else:
            self.last_location = current_location
            self.last_move_time = current_time

        return True
    
    def __enter__(self):
        
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

        for vehicle in self.world.get_actors().filter('vehicle.*'):
            if vehicle.id != self.ego_vehicle_id:
                vehicle.apply_control(carla.VehicleControl(throttle=0.05, brake=0.0))
        
        for _ in range(10):
            self.world.tick()
            time.sleep(0.05)
        
        dynamic_ids = set(v.id for v in self.world.get_actors().filter('vehicle.*'))
        
        self.collect_static_vehicles(dynamic_ids)

        return self
    

    def collect_static_vehicles(self, dynamic_ids=None):
        """get_level_bbs with error handling"""
        self.static_vehicles = []
        if dynamic_ids is None:
            dynamic_ids = set()
            dynamic_vehicles = self.world.get_actors().filter('vehicle.*')
            dynamic_ids = set(v.id for v in dynamic_vehicles)

        vehicle_types = [
            carla.CityObjectLabel.Car,
            carla.CityObjectLabel.Truck, 
            carla.CityObjectLabel.Bus,
            carla.CityObjectLabel.Motorcycle,
            carla.CityObjectLabel.Bicycle
        ]

        dynamic_vehicles = []
        dynamic_locations = []
        for dyn_id in dynamic_ids:
            try:
                dyn_vehicle = self.world.get_actor(dyn_id)
                if dyn_vehicle:
                    dynamic_vehicles.append(dyn_vehicle)
                    dynamic_locations.append(dyn_vehicle.get_location())
            except:
                pass

        for vehicle_type in vehicle_types:
            try:
                # get_level_bbs 
                print(f"Collecting {vehicle_type} (int: {int(vehicle_type)})")
                
                bounding_boxes = None
                for attempt in range(3):
                    try:
                        bounding_boxes = self.world.get_level_bbs(vehicle_type)
                        print(f"Successfully collected {len(bounding_boxes)} {vehicle_type} objects")
                        break
                    except Exception as e:
                        print(f"get_level_bbs attempt {attempt + 1} failed: {e}")
                        if attempt < 2:
                            time.sleep(0.5)  
                        else:
                            print(f"Failed to collect {vehicle_type} after 3 attempts, skipping...")
                            bounding_boxes = []
                            break
                
                if bounding_boxes is None:
                    bounding_boxes = []

                filtered_count = 0
                
                for i, bb in enumerate(bounding_boxes):
                    try:
                        # dynamic 차량과의 거리
                        is_dynamic = False
                        for dyn_loc in dynamic_locations:
                            if bb.location.distance(dyn_loc) < 1.5:
                                is_dynamic = True
                                filtered_count += 1
                                break
                        
                        if is_dynamic:
                            continue
                        
                        # Static 차량 수집
                        static_vehicle = {
                            'id': f"static_{vehicle_type}_{i}",
                            'type': int(vehicle_type),
                            'type_name': str(vehicle_type),
                            'transform': carla.Transform(bb.location, bb.rotation),
                            'bounding_box': bb,
                            'vertices': [v for v in bb.get_world_vertices(carla.Transform())],
                            'is_map_static': True 
                        }
                        self.static_vehicles.append(static_vehicle)
                    except Exception as e:
                        print(f"Error processing static vehicle {i}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Critical error collecting {vehicle_type}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"Total static vehicles collected: {len(self.static_vehicles)}")

    def get_carla_gt_3d_bboxes(self):        
        gt_bboxes = BoundingBox.get_lidar_based_3d_bboxes(
            self.world, 
            self.ego_vehicle_id, 
            self.static_vehicles, 
            self.object_tracker,
            max_distance=150.0
        )
        
        # print(f"Initial bbox count: {len(gt_bboxes)}")
        
        if self.semantic_lidar_sensor and self.current_semantic_data is not None:            
            try:
                semantic_detections = MultiOT128Processor.process_semantic_lidar_data(
                    self.current_semantic_data
                )
            
                camera_sensors = {}
                
                for sensor_dict in self.data_dict:
                    if sensor_dict["type"] == "sensor.camera.rgb":
                        camera_id = sensor_dict["id"]
                        camera_sensors[camera_id] = sensor_dict["sensor_spawned"]
            
                filtered_bboxes = Filtering.filter_bboxes_by_semantic_and_depth(
                    gt_bboxes, 
                    semantic_detections,
                    camera_sensors,
                    self.current_depth_data
                )
                
                gt_bboxes = filtered_bboxes
                # print(f"After improved filtering: {len(gt_bboxes)} bboxes")
                
            except Exception as e:
                print(f"Error in improved filtering: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to basic filtering...")
                
                semantic_detections = MultiOT128Processor.process_semantic_lidar_data(
                    self.current_semantic_data
                )
                filtered_bboxes = Filtering.filter_bboxes_by_semantic_lidar(
                    gt_bboxes, 
                    semantic_detections
                )
                gt_bboxes = filtered_bboxes
        else:
            if not self.semantic_lidar_sensor:
                print("No semantic LiDAR sensor available")
            else:
                print("No semantic LiDAR data available")
        
        return gt_bboxes
                
    def _save_basic_ai_annotation(self, bbox_list):
        try:
            annotator = get_basicai_annotator()
            
            if self.camera_configs and self.primary_lidar and self.camera_config_save_path:
                camera_config_list = annotator.process_all_cameras_for_basicai(
                    self.camera_configs, 
                    self.primary_lidar, 
                    self.frame_counter
                )
                
                if camera_config_list:
                    os.makedirs(self.camera_config_save_path, exist_ok=True)
                    camera_config_file = os.path.join(self.camera_config_save_path, f'{self.frame_counter:08d}.json')
                    with open(camera_config_file, 'w') as f:
                        json.dump(camera_config_list, f, indent=4)
                    print(f"Frame {self.frame_counter}: Saved camera config for {len(camera_config_list)} cameras")
            
            # BasicAI Annotation
            lidar_transform = self.primary_lidar.get_transform() if self.primary_lidar else None
            annotator.create_annotation(
                bbox_list,
                self.basic_ai_save_path,
                self.frame_counter,
                lidar_transform,
                self.track_uuid_manager,
                None,
                None,
                self.basic_ai_source_id,
                self.basic_ai_source_name
            )
            
        except Exception as e:
            print(f"Failed to save BasicAI annotation: {e}")
            print("Please check if ontology.json file is properly configured")
            import traceback
            traceback.print_exc()

    def tick(self, timeout=2.0):
        """Synchronous tick of all sensors."""

        # self.frame = self.world.tick()

        # World tick - 실패 시 즉시 예외
        try:
            self.frame = self.world.tick()
        except RuntimeError as e:
            if "std::exception" in str(e):
                print(f"World tick failed: {e}")
                print("WORLD_TICK_FAILED")  
                sys.stdout.flush()
                raise RuntimeError(f"World tick failed: {e}")
            else:
                raise e

        raw = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in raw)
        
        timestamp_us = int(time.time()*1e6)
        sample_token  = self.nusc.new_sample(timestamp_us, self.scene_token)
        ego_pose_tok  = self.nusc.add_ego_pose(self.vehicle.get_transform(), timestamp_us)

        self.frame_counter += 1

        rgb_camera_data = {}
        rgb_camera_sensors = {}
        depth_camera_data = {}
        
        # Multi-LiDAR data collection
        lidar_frame_data = {}
        
        nuscenes_annotations = []
        self.current_semantic_data=None
        self.current_depth_data = {}        # DEPTH DATA for bbox filtering
        
        # 1 : collecting Semantic LiDAR
        for sensor_dict, raw_data in zip(self.data_dict, raw[1:]):
            if sensor_dict["type"] == "sensor.lidar.ray_cast_semantic":
                self.current_semantic_data = raw_data
                break

        # 2: Collecting Depth data
        for sensor_dict, raw_data in zip(self.data_dict, raw[1:]):
            if sensor_dict["type"] == "sensor.camera.depth":
                depth_array = SensorProcessor.process_depth(raw_data)
                camera_id = sensor_dict["id"]  # ex: "depth_camera0"
                
                # print(f"Processing depth sensor: {camera_id}")
                # print(f"Depth array shape: {depth_array.shape}")
                # print(f"Depth range: {depth_array.min():.3f} - {depth_array.max():.3f}")
                
                # Key mapping : depth_camera0 → depth_camera0
                self.current_depth_data[camera_id] = depth_array
                
                # for viz
                mapped_camera_id = camera_id.replace('depth_', '')  # depth_camera0 → camera0
                depth_camera_data[mapped_camera_id] = depth_array

                # save depth data
                if sensor_dict.get("save_path"):
                    visualization_depth = (depth_array * 100) / 256
                    raw_path = os.path.join(sensor_dict["save_path"], 'raw', f'{self.frame_counter:08d}.png')
                    cv2.imwrite(raw_path, visualization_depth.astype(np.uint8))

        # Bbox 계산 (Semantic & Depth data 이미 수집되어있음)
        carla_gt_bboxes = self.get_carla_gt_3d_bboxes()
        
        # 1. 라이다 Range 기반 Filtering
        if carla_gt_bboxes:
            lidar_range = 200.0 
            for sensor_dict in self.data_dict:
                if sensor_dict["type"] == "sensor.lidar.ray_cast":
                    lidar_range = float(sensor_dict.get("range", 200.0))
                    break
            
            bbox_3d_coords = np.array([bbox_info for bbox_info in carla_gt_bboxes if bbox_info is not None], dtype=object)
            
            # simple distance filtering for lidar data
            # 이거 소용없는거 같은데 지울까말까
            if len(bbox_3d_coords) > 0:
                distances = []
                for bbox in bbox_3d_coords:
                    if isinstance(bbox, dict) and 'center_3d' in bbox:
                        # bbox format : 얓ㅅ
                        center = bbox['center_3d']
                        distance = np.linalg.norm(center)
                    else:
                        # bbox format : array
                        try:
                            bbox_array = np.array(bbox, dtype=np.float32)
                            if bbox_array.size >= 4:  # [class_id, x, y, z, ...]
                                distance = np.linalg.norm(bbox_array[1:4])
                            else:
                                distance = float('inf')  # 제외
                        except:
                            distance = float('inf')  # 제외
                    distances.append(distance)
                
                distances = np.array(distances)
                mask = distances <= lidar_range
                carla_gt_bboxes = bbox_3d_coords[mask].tolist()
                
           
        self._current_bboxes = []
        for bbox in carla_gt_bboxes:
            bbox_copy = bbox.copy() 
            self._current_bboxes.append(bbox_copy)
        
        #self.save_basic_ai_format(carla_gt_bboxes)
        self._save_basic_ai_annotation(carla_gt_bboxes)


        # NuScenes : world 좌표계로 변환
        for bbox in carla_gt_bboxes:
            class_id = bbox['class_id']
            center_xyz = [bbox['center_3d'][1], -bbox['center_3d'][0], bbox['center_3d'][2]]  # CARLA -> NuScenes 좌표계
            size_xyz = [bbox['size_3d'][1], bbox['size_3d'][0], bbox['size_3d'][2]]  # extent -> size
            yaw_rad = -np.deg2rad(bbox['yaw'])  # CARLA yaw -> NuScenes yaw
            nuscenes_annotations.append((class_id, center_xyz, size_xyz, yaw_rad))
        
        for sensor_dict, raw_data in zip(self.data_dict, raw[1:]):
            sensor_dict["raw_data"] = raw_data
                
            if sensor_dict["type"] == "sensor.camera.rgb":
                rgb_camera_data[sensor_dict["id"]] = raw_data
                rgb_camera_sensors[sensor_dict["id"]] = sensor_dict["sensor_spawned"]
            
            elif sensor_dict["type"] == "sensor.camera.depth":
                depth_array = SensorProcessor.process_depth(raw_data)
                camera_id = sensor_dict["id"].replace('depth_', '')
                depth_camera_data[camera_id] = depth_array

                visualization_depth = (depth_array * 100) / 256
                raw_path = os.path.join(sensor_dict["save_path"], 'raw', f'{self.frame_counter:08d}.png')
                cv2.imwrite(raw_path, visualization_depth.astype(np.uint8))
            
            elif sensor_dict["type"] == "sensor.camera.semantic_segmentation":
                array = SensorProcessor.process_segmentation(raw_data)
                
                raw_path = os.path.join(sensor_dict["save_path"], 'raw', f'{self.frame_counter:08d}.png')
                cv2.imwrite(raw_path, array)
                
                color_array = LABEL_COLORS[array]
                color_array = (color_array * 255).astype(np.uint8)
                color_array = cv2.cvtColor(color_array, cv2.COLOR_RGB2BGR)
                
                label_color_path = os.path.join(sensor_dict["save_path"], 'label_color', f'{self.frame_counter:08d}.png')
                cv2.imwrite(label_color_path, color_array)

            elif sensor_dict["type"] == "sensor.lidar.ray_cast":
                sensor_id = sensor_dict["id"]
                
                if raw_data is not None:
                    try:
                        current_lidar_transform = sensor_dict["sensor_spawned"].get_transform()
                        
                        points, intensity = MultiOT128Processor.lidar_callback(
                            raw_data, 
                            sensor_id, 
                            current_lidar_transform
                        )
                        
                        sensor_dict["processed_data"] = points
                        lidar_frame_data[sensor_id] = {'points': points, 'intensity': intensity}
                        
                    except Exception as e:
                        print(f"[{sensor_id}] LiDAR 처리 중 오류 발생: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Semantic LiDAR skip
            elif sensor_dict["type"] == "sensor.lidar.ray_cast_semantic":
                continue  

        # Multi Lidar merge processing
        try:
            combined_points, combined_intensity = MultiOT128Processor.merge_lidar_data()
            
            if combined_points is not None and len(combined_points) > 0:
                MultiOT128Processor.update_open3d(combined_points, combined_intensity, self._current_bboxes)
                combined_lidar_dir = os.path.join(self.save_root, 'sensor.lidar.ray_cast', 'combined_OT128')
                
                viz_dirs = {
                    'viz_top': os.path.join(combined_lidar_dir, 'viz', 'top'),
                    'viz_bbox': os.path.join(combined_lidar_dir, 'viz', 'bbox'),
                    'pcd': os.path.join(combined_lidar_dir, 'pcd'),
                    'bin': os.path.join(combined_lidar_dir, 'bin')
                }
                
                for dir_path in viz_dirs.values():
                    os.makedirs(dir_path, exist_ok=True)
                
                # viz
                MultiOT128Processor.save_current_open3d_frame(viz_dirs['viz_top'], self.frame_counter)
                MultiOT128Processor.save_bbox_overlay_frame(viz_dirs['viz_bbox'], self.frame_counter)
                
                # save pcd, bin
                MultiOT128Processor.save_combined_pcd(
                    viz_dirs['pcd'], 
                    self.frame_counter, 
                    combined_points, 
                    combined_intensity
                )
                
                MultiOT128Processor.save_combined_bin(
                    viz_dirs['bin'], 
                    self.frame_counter, 
                    combined_points, 
                    combined_intensity
                )
                
                if self.frame_counter % 10 == 0:
                    MultiOT128Processor.print_sensor_stats()
                                
        except Exception as e:
            print(f"Combined OT128 processing error: {e}")
            import traceback
            traceback.print_exc()

        # RGB camera bbox viz
        vehicles = self.world.get_actors().filter('vehicle.*')
        pedestrians = self.world.get_actors().filter('walker.pedestrian.*')
        moving_vehicles = [vehicle for vehicle in vehicles if vehicle.id != self.ego_vehicle_id]

        for sensor_dict, raw_data in zip(self.data_dict, raw[1:]):
            if sensor_dict["type"] == "sensor.camera.rgb":
                bbox_3d_coords = []
                static_infos = [] 
                
                depth_data = depth_camera_data.get(sensor_dict["id"], None)
                
                # 1) dynamic vehicle bbox (for viz)
                if moving_vehicles:
                    for vehicle in moving_vehicles:
                        bbox_result = BoundingBox.get_bounding_box(
                            vehicle, 
                            self.ego_vehicle_id, 
                            sensor_dict["sensor_spawned"], 
                            depth_data,  
                            occluded_check=True
                        )

                        if bbox_result is not None:
                            bbox_3d_coords.append(bbox_result)
                
                # 2) pedestrian bbox (for viz)
                if pedestrians:
                    for pedestrian in pedestrians:
                        bbox_result = BoundingBox.get_bounding_box(
                            pedestrian, 
                            self.ego_vehicle_id, 
                            sensor_dict["sensor_spawned"],
                            depth_data,  
                            occluded_check=True
                        )

                        if bbox_result is not None:
                            bbox_3d_coords.append(bbox_result)

                # 3) static vehicle bbox (for viz)
                processed_count = 0
                failed_count = 0
                for static_vehicle in self.static_vehicles:
                    if static_vehicle['is_map_static'] and static_vehicle['bounding_box'] is not None:
                        static_bbox_result = BoundingBox.get_environment_vehicle_bbox(
                            static_vehicle, 
                            sensor_dict["sensor_spawned"],
                            depth_data,
                            occluded_check=True,
                            coord="world"  # camera viz  : world
                        )

                        # save viz
                        if static_bbox_result is not None:
                            static_infos.append(static_bbox_result)
                            processed_count += 1
                        else:
                            failed_count += 1
                            
                # save bbox (for viz)
                sensor_dict['bbox_3d_coords'] = bbox_3d_coords
                sensor_dict['static_infos'] = static_infos

            if sensor_dict["type"] == "sensor.camera.rgb":
                array = SensorProcessor.process_rgb(raw_data, is_thermal=(sensor_dict["id"] == "keti_camera9_thermal"))
                raw_path = os.path.join(sensor_dict["save_path"], 'raw', f'{self.frame_counter:08d}.png')
                cv2.imwrite(raw_path, array)
                
                # bbox viz 
                all_bboxes = []
                for bbox in sensor_dict.get('bbox_3d_coords', []):
                    if bbox is not None:
                        all_bboxes.append(bbox)
                for static_info in sensor_dict.get('static_infos', []):
                    if static_info is not None:
                        all_bboxes.append(static_info)
                if "bbox_save_path" in sensor_dict:
                    viz_bbox_dir = os.path.join(sensor_dict["bbox_save_path"], 'viz_bbox')
                    os.makedirs(viz_bbox_dir, exist_ok=True)

                sensor_dict['frame_id'] = self.frame_counter

        # NuScenes 
        sensor_files = {}
        for sensor_dict in self.data_dict:
            if sensor_dict["type"] == "sensor.camera.rgb":
                rel_path = f"sensor.camera.rgb/{sensor_dict['id']}/raw/{self.frame_counter:08d}.png"
                sensor_files[sensor_dict["id"]] = rel_path
        
        rel_path = f"sensor.lidar.ray_cast/combined_OT128/bin/{self.frame_counter:08d}.bin"
        sensor_files["combined_OT128"] = rel_path
        
        self.nusc.add_sample(sensor_files)
        self.nusc.add_annotations(nuscenes_annotations)

        current_dynamic_ids = set(v.id for v in moving_vehicles)
        self.vehicle_ids = current_dynamic_ids
        return self.data_dict
    
    def __exit__(self, *args, **kwargs):
        """Reset settings when exiting context manager."""
        self.world.apply_settings(self._settings)
        self.nusc.finalize()

    def _retrieve_data(self, sensor_queue, timeout):
        """Retrieve data from a queue."""
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data