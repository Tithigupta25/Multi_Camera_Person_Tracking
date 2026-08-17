import cv2
import numpy as np

class ZoneManager:

    def __init__(self, zones=None):
        self.zones = zones or {}

    def point_in_polygon(self, point, polygon):

        if not polygon:
            return False

        polygon_points = np.array(polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(polygon_points, point, False)

        return result >= 0

    def get_zone(self, bbox):

        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        bottom_x = center_x
        bottom_y = int(y2)
        center_point = (center_x, center_y)
        bottom_point = (bottom_x, bottom_y)

        for zone_name, polygon in self.zones.items():

            if (self.point_in_polygon(center_point, polygon) or self.point_in_polygon(bottom_point, polygon)):
                return zone_name

        return None