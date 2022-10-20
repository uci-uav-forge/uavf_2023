import json

TAKEOFF = 22
WAYPOINT = 16
LAND = 21

class MissionObject:
    def __init__(self, home, waypoints, land):
        self.obj = dict()
        self.obj["cruiseSpeed"] = 15
        self.obj["firmwareType"] = 12
        self.obj["hoverSpeed"] = 5
        self.obj["items"] = list()

        self.obj["items"].append(self._generate_subdict(home, mode='TAKEOFF', doJumpId=1))

        for i in range(len(waypoints)):
            to_append = self._generate_subdict(waypoints[i], mode='WAYPOINT', doJumpId=i+2)
            self.obj["items"].append(to_append)
        
        self.obj["items"].append(self._generate_subdict(land, mode='LAND', doJumpId=len(waypoints)+2))

        self.obj["plannedHomePosition"] = (home[0], home[1]) + (home[2] / 2,)

        self.obj["vehicleType"] = 2
        self.obj["version"] = 2
        
    def export(self, path, json_type=False):
        if json_type == True:
            path += '.json'
        else:
            path += '.plan'
        template = open("mission_file_template.plan")
        template_json = json.load(template)
        template_json["mission"] = self.obj
        new_file = open(path, "w")
        json.dump(template_json, new_file)
        new_file.close()
        template.close()

    def _generate_subdict(self, coords, mode, doJumpId):
        to_append = dict()
        to_append["AMSLAltAboveTerrain"] = None
        to_append["Altitude"] = coords[2]
        to_append["AltitudeMode"] = 1
        to_append["autoContinue"] = True

        if mode == 'TAKEOFF':
            to_append["command"] = TAKEOFF
        if mode == 'LAND':
            to_append["command"] = LAND
        if mode == 'WAYPOINT':
            to_append["command"] = WAYPOINT

        to_append["doJumpId"] = doJumpId
        to_append["frame"] = 3
        to_append["params"] = [0, 0, 0, None]

        to_append["params"] += coords

        to_append["type"] = "SimpleItem"

        return to_append

if __name__ == '__main__':

    # MissionObject constructor takes list of GPS coordinates as input
    '''
    test = MissionObject(
        home=(33.64678917, -117.8426853, 50),
        waypoints=[
            (33.6466532, -117.84250671, 50),
            (33.64649436, -117.84254029, 50),
            (33.64647022, -117.84279674, 50)
        ],
        land=(33.6466316, -117.84288985, 0)
    )
    '''
    test = MissionObject(
        home=(47.3977507, 8.5456073, 100),
        waypoints=[
            (47.39777838174398, 8.544492608041622, 50),
            (47.39902740038751, 8.542841123072154, 60),
            (47.39967503756078, 8.547245178153947, 110),
            (47.39732731494636, 8.545570793020584, 80)
        ],
        land=(47.3977507, 8.5456073, 0)
    )
    # export function creates '.plan' file that can be uploaded to QGC
    # json_type MUST BE FALSE for QGC to read, enable for testing
    #test.export("generated_aldrich_path", json_type=False)
    test.export("default_path", json_type=False)