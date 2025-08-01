import numpy as np
import os
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
import carla
from datetime import datetime

VIRIDIS = np.array(matplotlib.colormaps['plasma'].colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

class MultiOT128Processor:
    
    vis = None
    point_list = None
    frame = 0
    ALPHA = 0.004       
    NORM_DIST = 200.0
    
    headless_mode = True
    current_points = None
    current_colors = None
    current_bboxes = None  
    
    # Store individual LiDAR data for merging
    lidar_data_buffer = {}
    lidar_transform = None  # Primary LiDAR transform for coordinate conversion
    
    sensor_stats = {}
    
    bbox_update_counter = 0
    last_bbox_frame = -1

    # def world_bbox_to_ot128_corrected(bbox_vertices_world, lidar_transform):
    #     """World → LiDAR 센서 좌표계 변환"""
    #     lidar_inv_matrix = np.array(lidar_transform.get_inverse_matrix())
        
    #     vertices_lidar = []
    #     for vertex in bbox_vertices_world:
    #         world_homo = np.array([vertex.x, vertex.y, vertex.z, 1.0])
    #         lidar_homo = lidar_inv_matrix @ world_homo
    #         vertices_lidar.append(lidar_homo[:3])
        
    #     vertices_lidar = np.array(vertices_lidar)
    #     vertices_lidar[:, 1] = -vertices_lidar[:, 1]  
        
    #     return vertices_lidar

    @staticmethod
    def setup_open3d(show_axis=False, headless=True):
        MultiOT128Processor.headless_mode = headless
        
        if headless:
            MultiOT128Processor.frame = 0
            MultiOT128Processor.bbox_update_counter = 0
            MultiOT128Processor.last_bbox_frame = -1
            return
        
        if MultiOT128Processor.vis is not None:
            return
        
        MultiOT128Processor.point_list = o3d.geometry.PointCloud()
        
        MultiOT128Processor.vis = o3d.visualization.Visualizer()
        MultiOT128Processor.vis.create_window(
            window_name='Multi OT128 Lidar (7 sensors combined)',
            width=1200,
            height=800,
            left=100,
            top=100)
        
        MultiOT128Processor.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        MultiOT128Processor.vis.get_render_option().point_size = 1
        MultiOT128Processor.vis.get_render_option().show_coordinate_frame = True
        
        if show_axis:
            MultiOT128Processor.add_open3d_axis(MultiOT128Processor.vis)
        
        MultiOT128Processor.frame = 0
    
    @staticmethod
    def add_open3d_axis(vis):
        axis = o3d.geometry.LineSet()
        axis.points = o3d.utility.Vector3dVector(np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]))
        axis.lines = o3d.utility.Vector2iVector(np.array([
            [0, 1],
            [0, 2],
            [0, 3]]))
        axis.colors = o3d.utility.Vector3dVector(np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]))
        vis.add_geometry(axis)
    
    @staticmethod
    def lidar_callback(data, sensor_id, sensor_transform=None):
        raw = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        xyz = raw[:, :3]
        intensity = raw[:, 3]

        corrected_xyz = np.column_stack((xyz[:, 0], -xyz[:, 1], xyz[:, 2]))
        
        # frame 당 LiDAR transform update
        if sensor_transform is not None:
            MultiOT128Processor.lidar_transform = sensor_transform
        
        if sensor_id not in MultiOT128Processor.sensor_stats:
            MultiOT128Processor.sensor_stats[sensor_id] = {'point_count': 0, 'frames': 0}
        
        MultiOT128Processor.sensor_stats[sensor_id]['point_count'] = len(corrected_xyz)
        MultiOT128Processor.sensor_stats[sensor_id]['frames'] += 1
        MultiOT128Processor.lidar_data_buffer[sensor_id] = {
            'points': corrected_xyz,
            'intensity': intensity,
            'timestamp': time.time(),
            'point_count': len(corrected_xyz)
        }
        
        return corrected_xyz, intensity
    
    @staticmethod
    def merge_lidar_data():
        """7개 LiDAR -> 1개 pcd(for hesai lidar) """
        if not MultiOT128Processor.lidar_data_buffer:
            return None, None
        
        all_points = []
        all_intensities = []
        total_points = 0
        
        sorted_sensors = sorted(MultiOT128Processor.lidar_data_buffer.keys())
        
        for sensor_id in sorted_sensors:
            data = MultiOT128Processor.lidar_data_buffer[sensor_id]

            point_count = data['point_count']
            all_points.append(data['points'])
            all_intensities.append(data['intensity'])
            total_points += point_count
        
        if all_points:
            combined_points = np.vstack(all_points)
            combined_intensity = np.hstack(all_intensities)
            distances = np.linalg.norm(combined_points, axis=1)
            valid_mask = distances <= 200.0
            combined_points = combined_points[valid_mask]
            combined_intensity = combined_intensity[valid_mask]

            MultiOT128Processor.lidar_data_buffer.clear()
            
            return combined_points, combined_intensity
        
        return None, None
    
    @staticmethod
    def process_semantic_lidar_data(semantic_measurement):
        detections = {}
        total_points = 0
        semantic_tags_found = set()
        
        try:
            # carla.SemanticLidarMeasurement에서 detection
            for detection in semantic_measurement:
                # carla.SemanticLidarDetection Object
                object_idx = detection.object_idx  
                semantic_tag = detection.object_tag
                
                if object_idx not in detections:
                    detections[object_idx] = semantic_tag
                
                total_points += 1
                semantic_tags_found.add(semantic_tag)
            
            unique_objects = len(detections)
        except Exception as e:
            print(f"Error processing semantic LiDAR data: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
        return detections    
    
    @staticmethod
    def update_open3d(xyz, intensity, bboxes=None):
        if xyz is None or len(xyz) == 0:
            return
            
        intensity_normalized = np.clip(intensity, 0, 1)
        
        int_color = np.c_[
            np.interp(intensity_normalized, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_normalized, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_normalized, VID_RANGE, VIRIDIS[:, 2])]

        if MultiOT128Processor.headless_mode:
            MultiOT128Processor.current_points = xyz.copy()
            MultiOT128Processor.current_colors = int_color.copy()
            
            if bboxes is not None:
                MultiOT128Processor.current_bboxes = []
                for bbox in bboxes:
                    bbox_copy = bbox.copy()
                    MultiOT128Processor.current_bboxes.append(bbox_copy)
                
                if MultiOT128Processor.frame != MultiOT128Processor.last_bbox_frame:
                    MultiOT128Processor.bbox_update_counter += 1
                    MultiOT128Processor.last_bbox_frame = MultiOT128Processor.frame
            else:
                MultiOT128Processor.current_bboxes = None
            
            MultiOT128Processor.frame += 1
            return

        if MultiOT128Processor.vis is None:
            MultiOT128Processor.setup_open3d(headless=False)
        
        # Open3D PCD update
        MultiOT128Processor.point_list.points = o3d.utility.Vector3dVector(xyz)
        MultiOT128Processor.point_list.colors = o3d.utility.Vector3dVector(int_color)
        MultiOT128Processor.frame += 1

        # bbox upload by frame
        MultiOT128Processor.current_bboxes = bboxes
        
        if MultiOT128Processor.frame == 2:
            MultiOT128Processor.vis.add_geometry(MultiOT128Processor.point_list)
        
        MultiOT128Processor.vis.update_geometry(MultiOT128Processor.point_list)
        MultiOT128Processor.vis.poll_events()
        MultiOT128Processor.vis.update_renderer()
        
        time.sleep(0.005)
    
    @staticmethod
    def save_current_open3d_frame(save_dir, frame_id):
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, f"{frame_id:08d}.png")
        
        if MultiOT128Processor.headless_mode:
            MultiOT128Processor.save_with_matplotlib(file_path, include_bboxes=False)
        else:
            if MultiOT128Processor.vis is not None:
                MultiOT128Processor.vis.capture_screen_image(file_path, do_render=True)
    
    @staticmethod
    def save_bbox_overlay_frame(save_dir, frame_id):
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, f"{frame_id:08d}.png")
        
        # if MultiOT128Processor.current_points is None:
        #     #print(f"Cannot save bbox overlay for frame {frame_id}: No point cloud data")
        #     return
            
        # # bbox상태확인
        # if MultiOT128Processor.current_bboxes is None:
        #     print(f"Frame {frame_id}: No bounding boxes to overlay")
        # else:
        #     print(f"Frame {frame_id}: Overlaying {len(MultiOT128Processor.current_bboxes)} bounding boxes")
            
        MultiOT128Processor.save_with_matplotlib(file_path, include_bboxes=True)
    
    @staticmethod
    def save_combined_pcd(save_dir, frame_id, points, intensity):
        os.makedirs(save_dir, exist_ok=True)
        
        # PCD 
        pcd_file = os.path.join(save_dir, f'{frame_id:08d}.pcd')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        intensity_normalized = np.clip(intensity, 0, 1)
        colors = np.c_[
            np.interp(intensity_normalized, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_normalized, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_normalized, VID_RANGE, VIRIDIS[:, 2])]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcd.estimate_normals()
        
        o3d.io.write_point_cloud(pcd_file, pcd)
    
    @staticmethod
    def save_combined_bin(save_dir, frame_id, points, intensity):
        os.makedirs(save_dir, exist_ok=True)
        
        bin_file = os.path.join(save_dir, f'{frame_id:08d}.bin')
        combined_data = np.column_stack((points, intensity))
        combined_data.astype(np.float32).tofile(bin_file)
            
    @staticmethod
    def save_with_matplotlib(file_path, include_bboxes=False):
        """pcd viz"""
        if MultiOT128Processor.current_points is None:
            return
            
        fig = plt.figure(figsize=(16, 12), facecolor='black', dpi=100)
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        
        points = MultiOT128Processor.current_points
        colors = MultiOT128Processor.current_colors
        
        if len(points) > 100000:
            indices = np.random.choice(len(points), 100000, replace=False)
            points_sampled = points[indices]
            colors_sampled = colors[indices]
        else:
            points_sampled = points
            colors_sampled = colors
        
        scatter = ax.scatter(points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2], 
                           c=colors_sampled, s=0.05, alpha=0.7)
        
        if include_bboxes:
            current_bboxes = MultiOT128Processor.current_bboxes
            if current_bboxes is not None and len(current_bboxes) > 0:
                MultiOT128Processor.draw_3d_bboxes(ax, current_bboxes)
            else:
                pass
        
        ax.view_init(elev=90, azim=0)
        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        ax.set_zlim([-10, 10])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.set_axis_off()
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(file_path, facecolor='black', edgecolor='none', 
                   bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        
        suffix = " with fresh bboxes" if include_bboxes else ""

    @staticmethod
    def world_to_lidar_point(world_point, lidar_transform):
        """World 좌표 -> LiDAR 센서 좌표"""
        lidar_inv_matrix = np.array(lidar_transform.get_inverse_matrix())
        
        world_homo = np.array([world_point.x, world_point.y, world_point.z, 1.0])
        lidar_homo = lidar_inv_matrix @ world_homo
        
        return lidar_homo[:3]  # x, y, z
    
    @staticmethod
    def draw_3d_bboxes(ax, bboxes):
        """3D 바운딩 박스를 matplotlib에 ㄱㄱ"""
        lidar_transform = getattr(MultiOT128Processor, 'lidar_transform', None)
        
        class_colors = {
            0: 'red',      # Vehicle
            1: 'green',    # Bicycle  
            2: 'blue',     # Pedestrian
        }
        
        bbox_count = 0
        
        if not isinstance(bboxes, list):
            return
            
            
        for i, bbox in enumerate(bboxes):
            try:
                if not isinstance(bbox, dict):
                    print(f"Bbox {i} is not a dict: {type(bbox)}")
                    continue
                
                if 'current_frame_transform' in bbox:
                    current_transform = bbox['current_frame_transform']
                    current_bbox = bbox.get('bbox_extent')
                    
                    if current_bbox and current_transform:
                        temp_bbox = carla.BoundingBox(
                            bbox.get('bbox_location', carla.Location(0,0,0)), 
                            current_bbox
                        )
                        vertices_world = temp_bbox.get_world_vertices(current_transform)
                    else:
                        vertices_world = bbox.get('vertices_world')
                else:
                    vertices_world = bbox.get('vertices_world')
                
                if vertices_world is None or len(vertices_world) != 8:
                    continue
                
                if lidar_transform is not None:
                    from bbox import Transform
                    vertices_corrected = Transform.world_to_lidar(
                        vertices_world, lidar_transform
                    )
                else:
                    vertices_corrected = np.array([[v.x, v.y, v.z] for v in vertices_world])
                
                class_id = bbox.get('class_id', 0)
                color = class_colors.get(class_id, 'white')
                
                MultiOT128Processor.draw_bbox_edges_correct(
                    ax, vertices_corrected, color, 1.5, 0.8
                )
                
                bbox_count += 1
                
            except Exception as e:
                import traceback
                traceback.print_exc()
    
    @staticmethod
    def draw_bbox_edges_correct(ax, vertices, color, linewidth, alpha=1.0):
        edges = [
            [0, 1], [1, 3], [3, 2], [2, 0],  # bottom
            [4, 5], [5, 7], [7, 6], [6, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7]   # sides
        ]
        
        for edge in edges:
            start_point = vertices[edge[0]]
            end_point = vertices[edge[1]]
            
            ax.plot3D([start_point[0], end_point[0]],
                     [start_point[1], end_point[1]],
                     [start_point[2], end_point[2]],
                     color=color, linewidth=linewidth, alpha=alpha)
    

    @staticmethod
    def print_sensor_stats():
        if MultiOT128Processor.sensor_stats:
            #print("\n=== Multi-LiDAR Sensor Statistics ===")
            total_points = 0
            for sensor_id, stats in MultiOT128Processor.sensor_stats.items():
                #print(f"{sensor_id}: {stats['point_count']:,} points/frame, {stats['frames']} frames")
                total_points += stats['point_count']
            #print(f"Total points per frame: {total_points:,}")
            #print(f"Bounding box updates: {MultiOT128Processor.bbox_update_counter}")
            #print("=====================================\n")
    
    @staticmethod
    def cleanup():
        if MultiOT128Processor.vis is not None:
            MultiOT128Processor.vis.destroy_window()
            MultiOT128Processor.vis = None
            MultiOT128Processor.point_list = None
            
        MultiOT128Processor.frame = 0
        MultiOT128Processor.current_points = None
        MultiOT128Processor.current_colors = None
        MultiOT128Processor.current_bboxes = None
        MultiOT128Processor.lidar_data_buffer.clear()
        MultiOT128Processor.sensor_stats.clear()
        
        MultiOT128Processor.bbox_update_counter = 0
        MultiOT128Processor.last_bbox_frame = -1
        

    @staticmethod
    def world_point_to_ot128_corrected(world_point, lidar_transform):
        lidar_inv_matrix = np.array(lidar_transform.get_inverse_matrix())
        
        world_homo = np.array([world_point.x, world_point.y, world_point.z, 1.0])
        lidar_homo = lidar_inv_matrix @ world_homo
        lidar_point = lidar_homo[:3]
        
        lidar_point[1] = -lidar_point[1]
        
        return lidar_point