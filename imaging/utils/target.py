class Target:
    def __init__(
        self, shape = None, 
        shape_color = None,
        letter = None, 
        letter_color = None, 
        center_coordinate = None, 
        gps_coordinate = None,
        bounding_box = None
    ) -> None:

        self.shape = shape
        self.shape_color = shape_color
        self.letter = letter
        self.letter_color = letter_color
        self.center_coordinate = center_coordinate
        self.gps_coordinate = gps_coordinate
        self.bounding_box = bounding_box
