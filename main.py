#!/usr/bin/env python

import os
import sys
os.environ['MPLBACKEND'] = 'Agg'

import glob
import json
import argparse
import time
import random
import datetime
import pygame
import carla
from lidar import MultiOT128Processor  # Updated import

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from carla_sync import CarlaSyncMode
from visualization import draw_images

# 차량생성
def spawn_hero_vehicle(world, json_data):
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.find(json_data["type"])
    
    if bp is None:
        raise RuntimeError(f"Could not find {json_data['type']} blueprint")
        
    # Set the vehicle ID/Role name
    bp.set_attribute('role_name', json_data["id"])
    
    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available")
    
    # Try spawning vehicle at random points until successful
    random.shuffle(spawn_points)
    vehicle = None
    
    for spawn_point in spawn_points:
        try:
            vehicle = world.spawn_actor(bp, spawn_point)
            if vehicle is not None:
                print(f"Successfully spawned hero vehicle {json_data['type']} with ID {json_data['id']} at {spawn_point}")
                return vehicle
        except Exception as e:
            print(f"Failed to spawn at point {spawn_points.index(spawn_point)}: {e}")
            continue
    
    if vehicle is None:
        raise RuntimeError(f"Failed to spawn hero vehicle {json_data['type']}")
    
    return vehicle

def setup_traffic_manager(client, vehicle, randomness=30.0, speed_limit=40.0):
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(0)
    vehicle.set_autopilot(True, tm_port)
    tm.set_desired_speed(vehicle, speed_limit)
    tm.set_percentage_random_deviation_speed(vehicle, randomness)
    tm.vehicle_percentage_speed_difference(vehicle, -10.0)  
    tm.auto_lane_change(vehicle, True)
    tm.set_offset(vehicle, 0.0)  
    tm.set_lane_change_probability(vehicle, 50.0)  
    tm.distance_to_leading_vehicle(vehicle, 5.0)
    tm.set_global_distance_to_leading_vehicle(5.0)
    tm.set_collision_detection(vehicle, True, any_vehicle=True)
    tm.ignore_lights_percentage(vehicle, 0)
    tm.ignore_signs_percentage(vehicle, 0)
    tm.vehicle_physics_control(vehicle, True)
        
    return tm

def main():
    global client
    
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '--objects_definition_file',
        default=os.path.join(os.getcwd(), "vehicle_config/ioniq_multilidar.json"),
        help='Vehicle definition file')
    argparser.add_argument(
        '--scale',
        metavar='SCALE',
        type=int,
        default='4',
        help='window scale (only integer)')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--time', type=int, default=30, help='Capture time limit in seconds (resets vehicle location)')
    argparser.add_argument('--stop', type=int, default=15, help='Vehicle stop time threshold in seconds')
    argparser.add_argument('--use_tm', action='store_true', default=True, 
                        help='Enable autonomous driving using Traffic Manager')
    argparser.add_argument('--fps', type=int, default=20,
                      help='Simulation FPS == LiDAR rotation_frequency')
    # NuScenes
    argparser.add_argument('--enable_nuscenes', action='store_true', default=True,
                          help='Enable NuScenes format export')
    argparser.add_argument('--scene_name', type=str, default='carla_scene',
                          help='Scene name for NuScenes export')

    args = argparser.parse_args()

    if not args.objects_definition_file or not os.path.exists(args.objects_definition_file):
        raise RuntimeError(
            "Could not read object definitions from {}".format(args.objects_definition_file))

    actor_list = []
    pygame.init()

    # JSON 파일에서 센서 정보 뽀려오기
    with open(args.objects_definition_file) as handle:
        json_vehicle = json.loads(handle.read())
    sensor_types = set()
    lidar_count = 0
    camera_count = 0
    
    for sensor in json_vehicle["sensors"]:
        if sensor["type"].startswith("sensor.camera"):
            sensor_type = sensor["type"].split(".")[-1]
            sensor_types.add(sensor_type)
            camera_count += 1
        elif sensor["type"].startswith("sensor.lidar"):
            sensor_types.add("lidar")
            lidar_count += 1
    
    print(f"Detected sensors: {camera_count} cameras, {lidar_count} LiDAR sensors")
    print(f"LiDAR sensors will be combined into one unified point cloud")
    
    num_rows = len(sensor_types)
    
    DISPLAY_WIDTH = 500
    DISPLAY_HEIGHT = 500
    
    display = pygame.display.set_mode(
        (DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    
    json_vehicle["detected_sensor_types"] = list(sensor_types)

    clock = pygame.time.Clock()

    print("Connecting to CARLA server...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)  
    
    world = client.get_world()
    
    print("Initializing Traffic Manager...")
    try:
        traffic_manager = client.get_trafficmanager(8000)  
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(0)
        print("Traffic Manager initialized successfully on port 8000")
    except Exception as e:
        print(f"Traffic Manager initialization failed: {e}")
        print("Continuing without Traffic Manager...")
        traffic_manager = None
    
    print("Setting up world synchronization...")
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    print(f"World synchronization set to {args.fps} FPS")

    # setup save path and folders
    date = datetime.datetime.today()
    save_date_name = f'{str(date.year)[-2:]}{date.month:02d}{date.day:02d}'
    base_save_dir = f'/home/keti/carlaUE5_data_demo/{save_date_name}_Carla_LidarFusion'
    save_dir_name = base_save_dir

    os.makedirs('merged', exist_ok=True)
    global_frame_counter = 1

    # Setup sensor save paths
    data_vehicle = json_vehicle["sensors"]
    for idx, sensor_spec in enumerate(data_vehicle):
        sensor_type = str(sensor_spec["type"])
        sensor_id = str(sensor_spec["id"])
        
        if sensor_type == "sensor.lidar.ray_cast":
            data_vehicle[idx]["save_path"] = None
            continue
        
        sensor_dir = os.path.join(save_dir_name, sensor_type, sensor_id)
        data_vehicle[idx]["save_path"] = sensor_dir
        
        if sensor_type == "sensor.camera.rgb":
            os.makedirs(os.path.join(sensor_dir, 'raw'), exist_ok=True)
        elif sensor_type == "sensor.camera.semantic_segmentation":
            os.makedirs(os.path.join(sensor_dir, 'raw'), exist_ok=True)
            os.makedirs(os.path.join(sensor_dir, 'label_color'), exist_ok=True)
        else:
            os.makedirs(os.path.join(sensor_dir, 'raw'), exist_ok=True)

    # Combined OT128 LiDAR directory
    combined_lidar_dir = os.path.join(save_dir_name, 'sensor.lidar.ray_cast', 'combined_OT128')
    viz_dirs = {
        'viz_top': os.path.join(combined_lidar_dir, 'viz', 'top'),
        'viz_bbox': os.path.join(combined_lidar_dir, 'viz', 'bbox'), 
        'pcd': os.path.join(combined_lidar_dir, 'pcd'),
        'bin': os.path.join(combined_lidar_dir, 'bin')
    }
    
    for dir_path in viz_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Created combined LiDAR directories for {lidar_count} sensors:")
    for name, path in viz_dirs.items():
        print(f"  {name}: {path}")

    # NuScenes 
    if args.enable_nuscenes:
        nuscenes_dir = os.path.join(save_dir_name, 'nuscenes')
        os.makedirs(nuscenes_dir, exist_ok=True)
        print(f"NuScenes export enabled. Data will be saved to: {nuscenes_dir}")

    global_start_time = time.time()

    try:
        while True:  # Main loop for respawning vehicle after timeout
            try:
                m = world.get_map()
                start_pose = random.choice(m.get_spawn_points())
                waypoint = m.get_waypoint(start_pose.location)

                print("Spawning hero vehicle...")
                vehicle = spawn_hero_vehicle(world, json_vehicle)
                if vehicle is None:
                    raise RuntimeError("Failed to spawn hero vehicle")

                vehicle.set_simulate_physics(True)

                # ========== Autopilot ==========
                autopilot_success = False
                if traffic_manager is not None:
                    print("Setting up autonomous driving...")
                    for attempt in range(5):
                        try:
                            print(f"  Attempt {attempt + 1}/5: Enabling autopilot...")
                            vehicle.set_autopilot(True, 8000)  # Traffic Manager port
                            
                            # Traffic Manager Setting
                            traffic_manager.set_desired_speed(vehicle, 30.0)
                            traffic_manager.set_global_distance_to_leading_vehicle(4.0)
                            traffic_manager.distance_to_leading_vehicle(vehicle, 4.0)
                            traffic_manager.vehicle_percentage_speed_difference(vehicle, -10.0)
                            
                            autopilot_success = True
                            break
                            
                        except RuntimeError as e:
                            if "timeout" in str(e).lower():
                                print(f"Timeout on attempt {attempt + 1}")
                                if attempt < 4:
                                    print("Waiting 3 seconds before retry...")
                                    time.sleep(3)
                                    continue
                            else:
                                print(f"Unexpected error: {e}")
                                break
                        except Exception as e:
                            print("Error setting autopilot: {e}")
                            break
                
                if not autopilot_success:
                    print("Failed to enable autopilot, using manual control")
                    vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))

                start_time = time.time()                

                # CarlaSyncMode context for sensors and simulation
                print("Starting CarlaSyncMode...")
                with CarlaSyncMode(world, json_vehicle["sensors"], vehicle, args.stop, client=client, 
                                save_root=save_dir_name, scene_name=f"{args.scene_name}_{global_frame_counter:04d}", 
                                fps=args.fps) as sync_mode:
                    actor_list.append(vehicle)
                    actor_list.extend(sync_mode.sensors)
    
                    sync_mode.frame_counter = global_frame_counter
                    MultiOT128Processor.setup_open3d(show_axis=False, headless=True)

                    # print(f"Multi-OT128 system initialized with {len(sync_mode.lidar_sensors)} LiDAR sensors")
                    # print(f"Expected combined point cloud: ~{lidar_count * 50000:,} points per frame")
                    # print(f"NuScenes exporter initialized for scene: {args.scene_name}_{global_frame_counter:04d}")

                    while True:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return True
                            elif event.type == pygame.KEYUP:
                                if event.key == pygame.K_ESCAPE:
                                    return True

                        clock.tick()
                        
                        # Check vehicle movement
                        if not sync_mode.check_vehicle_movement():
                            print("VEHICLE_STOPPED - Respawning vehicle")
                            MultiOT128Processor.print_sensor_stats()
                            global_frame_counter = sync_mode.frame_counter
                            break

                        # Advance simulation and wait for data
                        data_dict = sync_mode.tick(timeout=5.0) 

                        # Update display
                        draw_images(display, data_dict, save_dir_name, sync_mode.frame_counter)
                        pygame.display.flip()

                        global_frame_counter = sync_mode.frame_counter
                        global_elapsed_time = int(time.time() - global_start_time)
                        global_remaining_time = max(0, args.time - global_elapsed_time)

                        # ETA
                        eta = datetime.datetime.now() + datetime.timedelta(seconds=global_remaining_time)
                        eta_str = eta.strftime('%H:%M:%S')
                        progress = (global_elapsed_time / args.time) * 100 if args.time > 0 else 100

                        print(f'Frame: {global_frame_counter:08d} | Progress: {progress:.1f}% | '
                            f'Global Elapsed: {datetime.timedelta(seconds=global_elapsed_time)} | '
                            f'Remaining: {datetime.timedelta(seconds=global_remaining_time)} | '
                            f'ETA: {eta_str}')
                        sys.stdout.flush()

                        
                        if global_elapsed_time >= args.time:
                            print(f"Global capture time limit ({args.time} seconds) reached. Exiting...")
                            MultiOT128Processor.print_sensor_stats()
                            return  

                        clock.tick(30)
                        
                # Clean up sensors and prepare for respawn
                print("Cleaning up sensors for respawn...")
                
                # Disable autopilot
                if vehicle and hasattr(vehicle, 'set_autopilot'):
                    try:
                        vehicle.set_autopilot(False)
                        print("Autopilot disabled")
                    except Exception as e:
                        print(f"Failed to disable autopilot: {e}")
                
                # Destroy all actors
                for actor in actor_list:
                    try:
                        actor.destroy()
                    except Exception as e:
                        print(f"Failed to destroy actor: {e}")
                actor_list.clear()
                
            except KeyboardInterrupt:
                print('\nCancelled by user. Bye!')
                break  # Exit program on keyboard interrupt
                
            except Exception as e:
                print(f"Error occurred: {e}")
                import traceback
                traceback.print_exc()
                
                # Clean up on error
                for actor in actor_list:
                    try:
                        if hasattr(actor, 'set_autopilot'):
                            actor.set_autopilot(False)
                        actor.destroy()
                    except:
                        pass
                actor_list.clear()
                
                print("Retrying after error...")
                
            # Wait before next spawn cycle
            print("Waiting 3 seconds before respawning...")
            time.sleep(3)

    finally:
        print('Restoring world settings and destroying actors...')
        
        # Restore original world settings
        try:
            world.apply_settings(original_settings)
            print("World settings restored")
        except Exception as e:
            print(f"Failed to restore world settings: {e}")
        
        # Destroy all actors
        for actor in actor_list:
            try:
                if actor and hasattr(actor, 'set_autopilot'):
                    actor.set_autopilot(False)
                actor.destroy()
            except Exception as e:
                print(f"Failed to destroy actor: {e}")
                
        try:
            MultiOT128Processor.cleanup()
            print("Final sensor statistics:")
            MultiOT128Processor.print_sensor_stats()
        except Exception as e:
            print(f"Cleanup error: {e}")
            
        pygame.quit()
        print('Done.')

if __name__ == '__main__':
    main()