class IIR2Filter:
    def __init__(self, coefficients):  # need array of 6 elements input
        self.b0 = coefficients[0]
        self.b1 = coefficients[1]
        self.b2 = coefficients[2]
        self.a1 = coefficients[4]
        self.a2 = coefficients[5]
        self.acc = 0;  # accumulator
        self.b_x1 = 0;  # input x delay buffer
        self.b_x2 = 0;
        self.b_y1 = 0;  # output y delay buffer
        self.b_y2 = 0;

    def dofilter(self, x):
        self.acc = (x * self.b0) + (self.b_x1 * self.b1) + (self.b_x2 * self.b2) - (self.b_y1 * self.a1) - (
                    self.b_y2 * self.a2)
        self.b_x2 = self.b_x1  # update delay lines
        self.b_x1 = x
        self.b_y2 = self.b_y1
        self.b_y1 = self.acc
        return self.acc


class IIRFilter:
    def __init__(self, coefficients_array):
        self.filters = []
        for i in range(len(coefficients_array)):
            self.filters.append(IIR2Filter(coefficients_array[i]))  # instantiate 2nd order filters

    def dofilter(self, u):
        for i in range(len(self.filters)):
            u = self.filters[i].dofilter(u)
        return u