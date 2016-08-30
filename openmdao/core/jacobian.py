
FWD, REV = 0, 1


# during setup, it's assumed that components that will provide us
# with a sparse jacobian will specify sparseness with COO'ish set of row and col indices
# for each pair of vars in the jacobian.
class CSRJacobian(object):
    def __init__(self, mode=FWD):
        self.mode = mode
        self.mat = None  # the CSR sparse matrix

    def set_sizes(self, sizes):
        pass
        """
        - based on mode, create mapping of (o,i) to row, col, offset, value_view
        - what if we're in rev mode and they give us a sparse J and we have to
            transpose it?
        """

    def __setitem__(self, output, input, value):
        pass

    def __getitem__(self, output, input):
        pass



if __name__ == '__main__':
    j = CSRJacobian()
    j.set_sizes()
