import json
import carla
import numpy as np
from typing import Dict, Optional, Any

class OntologyMapper:
    def __init__(self, ontology_file_path: str):
        self.ontology_file_path = ontology_file_path
        self.class_mapping = {}  # CARLA class_id -> ontology info
        self.class_values_mapping = {}  # classId -> attribute template
        self.semantic_value_mapping = {}  # semantic_tag -> value string
        
        self._load_ontology()
        self._build_mappings()
    
    def _load_ontology(self):
        try:
            with open(self.ontology_file_path, 'r', encoding='utf-8') as f:
                self.ontology_data = json.load(f)
            print(f"Ontology loaded from: {self.ontology_file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Ontology file not found: {self.ontology_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in ontology file: {e}")
    
    def _build_mappings(self):
        classes = self.ontology_data.get("classes", [])
        
        # 다른 Ontology class도 추가;;; 근데 Carla에서 동물 소환 못할텐데 ㅋ
        four_wheels_class = self._find_class_by_name("Four_Wheels")
        two_wheels_class = self._find_class_by_name("Two_Wheels")  
        pedestrian_class = self._find_class_by_name("Pedestrian")
        
        if not all([four_wheels_class, two_wheels_class, pedestrian_class]):
            missing = []
            if not four_wheels_class: missing.append("Four_Wheels")
            if not two_wheels_class: missing.append("Two_Wheels")
            if not pedestrian_class: missing.append("Pedestrian")
            raise ValueError(f"Required classes not found in ontology: {missing}")
        
        # Mapping : CARLA class_id -> ontology
        self.class_mapping = {
            0: {"classId": four_wheels_class["id"], "className": "Four_Wheels"},
            1: {"classId": two_wheels_class["id"], "className": "Two_Wheels"},
            2: {"classId": pedestrian_class["id"], "className": "Pedestrian"}
        }
        
        self._build_class_values_mapping(four_wheels_class)
        self._build_class_values_mapping(two_wheels_class)
        self._build_class_values_mapping(pedestrian_class)
        
        self._build_semantic_value_mapping()
        
        print(f"Ontology mappings built successfully:")
        print(f"  - Four_Wheels (classId: {self.class_mapping[0]['classId']})")
        print(f"  - Two_Wheels (classId: {self.class_mapping[1]['classId']})")
        print(f"  - Pedestrian (classId: {self.class_mapping[2]['classId']})")
    
    def _find_class_by_name(self, class_name: str) -> Optional[Dict]:
        for cls in self.ontology_data.get("classes", []):
            if cls.get("name") == class_name:
                return cls
        return None
    
    def _build_class_values_mapping(self, class_info: Dict):
        class_id = class_info["id"]
        attributes = class_info.get("attributes", [])
        
        if not attributes:
            print(f"Warning: No attributes found for class {class_info['name']}")
            return
        
        primary_attribute = attributes[0]
        
        self.class_values_mapping[class_id] = {
            "attributeVersion": primary_attribute.get("version", 1),
            "id": primary_attribute["id"],
            "isLeaf": True,
            "name": primary_attribute["name"],
            "type": primary_attribute["type"]
        }
    
    def _build_semantic_value_mapping(self):
        four_wheels_class = self._find_class_by_name("Four_Wheels")
        if four_wheels_class and four_wheels_class.get("attributes"):
            vehicle_options = self._get_attribute_options(four_wheels_class, "Vehicle")
            if vehicle_options:
                option_names = [opt.get("name", "") for opt in vehicle_options]
                
                self.semantic_value_mapping.update({
                    int(carla.CityObjectLabel.Car): "car" if "car" in option_names else option_names[0] if option_names else "car",
                    int(carla.CityObjectLabel.Truck): "truck" if "truck" in option_names else "car",
                    int(carla.CityObjectLabel.Bus): "bus" if "bus" in option_names else "car",
                })
        
        # Two_Wheels (Vehicle) 세부 분류  
        two_wheels_class = self._find_class_by_name("Two_Wheels")
        if two_wheels_class and two_wheels_class.get("attributes"):
            vehicle_options = self._get_attribute_options(two_wheels_class, "Vehicle")
            if vehicle_options:
                option_names = [opt.get("name", "") for opt in vehicle_options]
                
                self.semantic_value_mapping.update({
                    int(carla.CityObjectLabel.Motorcycle): "motorcycle" if "motorcycle" in option_names else option_names[0] if option_names else "bicycle",
                    int(carla.CityObjectLabel.Bicycle): "bicycle" if "bicycle" in option_names else option_names[0] if option_names else "bicycle"
                })
        
        # Pedestrian (Human) - 보통 고정값
        pedestrian_class = self._find_class_by_name("Pedestrian")
        if pedestrian_class and pedestrian_class.get("attributes"):
            human_options = self._get_attribute_options(pedestrian_class, "Human")
            if human_options:
                option_names = [opt.get("name", "") for opt in human_options]
                default_human_value = "adult" if "adult" in option_names else option_names[0] if option_names else "adult"
            else:
                default_human_value = "adult"
        else:
            default_human_value = "adult"
            
        self.semantic_value_mapping[int(carla.CityObjectLabel.Pedestrians)] = default_human_value
    
    def _get_attribute_options(self, class_info: Dict, attribute_name: str) -> Optional[list]:
        """클래스의 특정 속성 옵션들 반환"""
        for attr in class_info.get("attributes", []):
            if attr.get("name") == attribute_name:
                return attr.get("options", [])
        return None
    
    def get_class_info(self, carla_class_id: int) -> Dict:
        return self.class_mapping.get(carla_class_id, self.class_mapping[0])
    
    def get_class_values(self, carla_class_id: int, semantic_tag: Optional[int] = None) -> Optional[list]:
        class_info = self.get_class_info(carla_class_id)
        class_id = class_info["classId"]
        
        # 속성 템플릿 가져오기
        template = self.class_values_mapping.get(class_id)
        if not template:
            print(f"Warning: No attribute template found for classId {class_id}")
            return None
        
        # Semantic tag 기반으로 value 결정
        if semantic_tag is not None and semantic_tag in self.semantic_value_mapping:
            value = self.semantic_value_mapping[semantic_tag]
        else:
            # 기본값 
            if carla_class_id == 0:  # Four_Wheels
                value = "car"
            elif carla_class_id == 1:  # Two_Wheels  
                value = "bicycle"
            elif carla_class_id == 2:  # Pedestrian
                value = "adult"
            else:
                value = "car"
        
        return [{
            "attributeVersion": template["attributeVersion"],
            "id": template["id"],
            "isLeaf": template["isLeaf"],
            "name": template["name"],
            "type": template["type"],
            "value": value
        }]
    
    def get_class_version(self, carla_class_id: int, is_static: bool = False) -> int:
        if carla_class_id == 0:  # Four_Wheels (Vehicle)
            if is_static:
                return 1  # Static vehicles는 주로 1
            else:
                # Dynamic vehicles는 1이 주류, 일부 4
                return int(np.random.choice([1, 4], p=[0.7, 0.3]))
        elif carla_class_id == 1:  # Two_Wheels
            return int(np.random.choice([1, 2], p=[0.5, 0.5]))
        elif carla_class_id == 2:  # Pedestrian
            return int(np.random.choice([1, 4], p=[0.85, 0.15]))
        else:
            return 1  # 기본값
    
    def get_ontology_summary(self) -> Dict:
        return {
            "file_path": self.ontology_file_path,
            "class_mappings": self.class_mapping,
            "semantic_mappings": self.semantic_value_mapping,
            "total_classes": len(self.ontology_data.get("classes", [])),
            "available_classes": [cls.get("name") for cls in self.ontology_data.get("classes", [])]
        }