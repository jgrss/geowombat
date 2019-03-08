class GeoProperties(object):

    @property
    def layers(self):
        self._get_shape()
        return self.layers_

    @property
    def rows(self):
        self._get_shape()
        return self.rows_

    @property
    def columns(self):
        self._get_shape()
        return self.cols_

    @property
    def left(self):
        return self.src.left

    @property
    def right(self):
        return self.src.right

    @property
    def top(self):
        return self.src.top

    @property
    def bottom(self):
        return self.src.bottom

    @property
    def extent(self):
        return [self.left, self.right, self.top, self.bottom]

    @property
    def crs(self):
        return self.src.projection

    @property
    def cell_y(self):
        return self.src.cellY

    @property
    def cell_x(self):
        return self.src.cellX

    @property
    def no_data(self):
        return self.no_data_
        
