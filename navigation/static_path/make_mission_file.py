import json

TAKEOFF = 22
WAYPOINT = 16
LAND = 21

class MissionObject:
    def __init__(self, waypoints):
        self.obj = dict()
        self.obj["cruiseSpeed"] = 15
        self.obj["firmwareType"] = 12
        self.obj["hoverSpeed"] = 5
        self.obj["items"] = list()

        for i in range(len(waypoints)):
            to_append = dict()
            to_append["AMSLAltAboveTerrain"] = None
            to_append["Altitude"] = waypoints[i][2]
            to_append["AltitudeMode"] = 1
            to_append["autoContinue"] = True

            if i == 0:
                to_append["command"] = TAKEOFF
            elif i == len(waypoints) - 1:
                to_append["command"] = LAND
            else:
                to_append["command"] = WAYPOINT

            to_append["doJumpId"] = 1
            to_append["frame"] = 3
            to_append["params"] = [0, 0, 0, None]

            to_append["params"] += waypoints[i]

            to_append["type"] = "SimpleItem"

            self.obj["items"].append(to_append)
        

        self.obj["plannedHomePosition"] = list(waypoints[0])

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

if __name__ == '__main__':
    test = MissionObject(
        [
            (39.848767191215124, -123.14216773753992, 50), 
            (43.04966623700112, -122.6836030911128, 50), 
            (40.23677990726233, -112.3538310558068, 50)])
    test.export("test1", json_type=False)