#!/bin/bash

LOG_FILE="carla_simulation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Cleanup function
cleanup() {
    echo "Cleaning up resources..."
    
    # Kill all background processes
    pkill -9 CarlaUnreal.sh
    echo "CarlaUnreal.sh killed"
    
    # Kill all xterm windows
    pkill -9 xterm
    echo "xterm processes killed"
    
    # Kill all Python processes
    pkill -9 python3
    pkill -9 python
    echo "Python processes killed"
    
    # Kill any remaining Pygame windows
    if pgrep -f "pygame" > /dev/null; then
        pkill -9 -f "pygame"
        echo "Pygame windows killed"
    fi
    
    # Clear memory cache if possible
    if [ "$EUID" -eq 0 ]; then
        echo 3 > /proc/sys/vm/drop_caches
        swapoff -a && swapon -a
        echo "Memory cache cleared"
    fi
    
    exit 0
}

# Set up trap for Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

# CARLA Configuration
CARLA_HOST_IP=127.0.0.1
CARLA_HOST_PORT=2000
CARLA_PATH=./Carla-0.10.0-Linux-Shipping
RUN_CARLA=true

# Simulation Configuration
EXPLORATION_TIME=600     
STOP_TIME=30            # 최대 정지시간 ~> spawn 초기화 후 다시 실행
VEHICLE_CONFIG=vehicle_config/ioniq_ideal.json
N_OF_VEHICLE=100
N_OF_PEDESTRIAN=100

# Ontology Configuration
ONTOLOGY_CONFIG_DIR="./ontology_config"
ONTOLOGY_FILE_PATH="${ONTOLOGY_CONFIG_DIR}/ontology.json"

# Function to check ontology file 
check_ontology() {
    echo "Checking ontology configuration..."
    
    if [ ! -d "$ONTOLOGY_CONFIG_DIR" ]; then
        echo "Creating ontology config directory: $ONTOLOGY_CONFIG_DIR"
        mkdir -p "$ONTOLOGY_CONFIG_DIR"
    fi
    
    if [ ! -f "$ONTOLOGY_FILE_PATH" ]; then
        echo "================================================================================================"
        echo " NOTICE: Ontology file not found at: $ONTOLOGY_FILE_PATH"
        echo ""
        echo " The simulation will continue with basic annotations (without ontology)."
        echo " If you want to use BasicAI ontology:"
        echo "   1. Export ontology.json from BasicAI annotation tool"
        echo "   2. Place it at: $ONTOLOGY_CONFIG_DIR/ontology.json"
        echo "   3. Restart this script"
        echo ""
        echo " For now, proceeding without ontology..."
        echo "================================================================================================"
        
        export ONTOLOGY_FILE_PATH=""
        echo "Continuing without ontology file."
        return 0
    else
        echo "Ontology file found: $ONTOLOGY_FILE_PATH"
        export ONTOLOGY_FILE_PATH="$ONTOLOGY_FILE_PATH"
        return 0
    fi
}

# Function to prompt user for ontology
prompt_ontology_choice() {
    echo ""
    echo "Would you like to specify an ontology file path? (y/n)"
    echo "Press Enter to continue without ontology, or 'y' to specify a path:"
    read -r user_choice
    
    case "$user_choice" in
        [Yy]*)
            echo "Please enter the full path to your ontology.json file:"
            read -r custom_ontology_path
            
            if [ -f "$custom_ontology_path" ]; then
                echo "Using ontology file: $custom_ontology_path"
                export ONTOLOGY_FILE_PATH="$custom_ontology_path"
                return 0
            else
                echo "File not found: $custom_ontology_path"
                echo "Continuing without ontology..."
                export ONTOLOGY_FILE_PATH=""
                return 0
            fi
            ;;
        *)
            echo "Continuing without ontology..."
            export ONTOLOGY_FILE_PATH=""
            return 0
            ;;
    esac
}

# Function to check server connection
check_server() {
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Checking server connection (attempt $attempt/$max_attempts)..."
        if python3 -c "
import carla
try:
    client = carla.Client('${CARLA_HOST_IP}', ${CARLA_HOST_PORT})
    client.set_timeout(10.0)
    world = client.get_world()
    print('Server is ready')
    exit(0)
except Exception as e:
    print(f'Server not ready: {e}')
    exit(1)
"; then
            return 0  # 성공하면 즉시 return
        fi
        
        attempt=$((attempt + 1))
        if [ $attempt -le $max_attempts ]; then
            echo "Waiting before next attempt..."
            sleep 5
        fi
    done
    return 1
}

# Function to generate traffic
generate_traffic() {
    echo "Generating traffic..."

    python3 ${CARLA_PATH}/PythonAPI/examples/generate_traffic.py \
        --host $CARLA_HOST_IP \
        --port $CARLA_HOST_PORT \
        -n $N_OF_VEHICLE \
        -w $N_OF_PEDESTRIAN \
        --safe \
        --respawn 2>/dev/null &
        
    sleep 10  
    
    # Traffic Manager 설정
    python3 -c "
import carla
import time
try:
    client = carla.Client('${CARLA_HOST_IP}', ${CARLA_HOST_PORT})
    client.set_timeout(10.0)
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_hybrid_physics_mode(True)
    print('Traffic Manager configured successfully')
except Exception as e:
    print(f'Failed to configure Traffic Manager: {e}')
    exit(1)
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "Traffic generation successful"
        return 0
    else
        echo "Failed to generate traffic"
        return 1
    fi
}

# Function to run CARLA
run_carla() {
    if [ "$RUN_CARLA" = true ]; then
        pip install numpy ${CARLA_PATH}/PythonAPI/carla/dist/carla-0.10.0-cp310-cp310-linux_x86_64.whl 2>/dev/null

        bash ${CARLA_PATH}/CarlaUnreal.sh --fps=20 &

        # Wait for CARLA to start
        echo "Waiting for CARLA to start..."
        sleep 10

        # Single connection check
        if check_server; then
            echo "CARLA server started successfully"
        else
            echo "Failed to start CARLA server"
            exit 1
        fi
    fi
}

# Main execution 
main() {
    # 기본 온톨로지 체크
    check_ontology
    
    # 온톨로지 파일이 없는 경우에만 사용자에게 선택권 제공
    if [ -z "$ONTOLOGY_FILE_PATH" ] || [ "$ONTOLOGY_FILE_PATH" = "" ]; then
        prompt_ontology_choice
    fi
    
    run_carla

    # Generate initial traffic
    if ! generate_traffic; then
        echo "Failed to generate initial traffic. Exiting..."
        cleanup
        exit 1
    fi

    # Server check
    if ! check_server; then
        echo "Lost connection to CARLA server. Attempting to restart..."
        cleanup
        run_carla
    fi
    
    # main.py 실행 (온톨로지 파일 경로 전달)
    echo "Starting main.py..."
    if [ -n "$ONTOLOGY_FILE_PATH" ] && [ "$ONTOLOGY_FILE_PATH" != "" ]; then
        echo "Using ontology: $ONTOLOGY_FILE_PATH"
    else
        echo "Running without ontology (basic annotations only)"
    fi
    
    ONTOLOGY_FILE_PATH="$ONTOLOGY_FILE_PATH" python3 main.py \
        --host $CARLA_HOST_IP \
        --port $CARLA_HOST_PORT \
        --time $EXPLORATION_TIME \
        --stop $STOP_TIME \
        --objects_definition_file $VEHICLE_CONFIG
        # 2>/dev/null
        
    # Wait before cleanup
    sleep 5
    
    cleanup
}

# Run the main function
main

echo "Cleaning up..."
pkill -9 CarlaUnreal.sh
pkill -9 xterm
pkill -9 python3
pkill -9 python

echo "Clearing memory cache"
if [ "$EUID" -eq 0 ]; then
    echo 3 > /proc/sys/vm/drop_caches
    swapoff -a && swapon -a
    echo "Memory cache cleared"
fi
