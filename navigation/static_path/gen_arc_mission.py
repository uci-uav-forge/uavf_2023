from make_mission_file import MissionObject
import sys
sys.path.insert(0, '/home/herpderk/uav_catkin_ws/src/uavf_2023/navigation/algorithms/global_path')
from flight_plan import Flight_Zone

bound_coords = [
    (33.641890, -117.825961),
    (33.641290, -117.826358),
    (33.640768, -117.825367),
    (33.641339, -117.824964)
]
start = (33.641362, -117.826097)    
wps = [
    (33.641659, -117.825890), 
    (33.641443, -117.825440),
    (33.641282, -117.825079),
    (33.641170, -117.825210),
    (33.641433, -117.825741),
    (33.641283, -117.825886),
    (33.641362, -117.826097)
]

arc_map = Flight_Zone(bound_coords)
arc_map.gen_globalpath(start, wps)
wp_order = arc_map.wp_order

arc_mission = MissionObject(
    home=(start[0], start[1], 30),
    waypoints=[
        (wp_order[0][0], wp_order[0][1], 30),
        (wp_order[1][0], wp_order[1][1], 30),
        (wp_order[2][0], wp_order[2][1], 30),
        (wp_order[3][0], wp_order[3][1], 30),
        (wp_order[4][0], wp_order[4][1], 30),
        (wp_order[5][0], wp_order[5][1], 30),
        (wp_order[6][0], wp_order[6][1], 30)
    ],
    land=(start[0], start[1], 0)
)
arc_mission.export("arc_field", json_type=False)