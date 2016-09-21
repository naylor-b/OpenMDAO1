import unittest

import numpy as np
from scipy.sparse import coo_matrix
from six import iteritems

from openmdao.core.jacobian import SparseJacobian

def dump(mat):
    cmat = mat.tocoo()
    for i,j,val in zip(cmat.row, cmat.col, cmat.data):
        print i,j,val
        
class TestSparseJacobian(unittest.TestCase):
    def setUp(self):
        np.set_printoptions(precision=2, suppress=True)
        dtype = int
        
        self.slices = {
            'a3': (12,15),
            'b2': (7,9),
            'c1': (3,4),
            'd3': (4,7),
            'x3': (0,3),
            'y1': (9,10),
            'z2': (10,12),
        }
        
        # in this J, a,b,c are outputs and x,y,z are inputs.  b is also a state
        self.subjacs = {
            ('x3','a3') : (np.array([[1,0,6],[5,5,9],[0,0,3]], dtype=dtype), None),
            ('y1','a3') : (np.array([[1,7,0]], dtype=dtype), None),
            ('z2','a3') : (np.array([[6,0],[3,5]], dtype=dtype), [2,0]),
            ('b2','b2') : (np.array([[0,8],[9,7]], dtype=dtype), None),
            ('x3','b2') : (np.array([[9,9],[0,8],[5,4]], dtype=dtype), None),
            ('y1','b2') : (np.array([[9,3]], dtype=dtype), None),
            ('z2','b2') : (np.array([[5,0],[0,2]], dtype=dtype), None),
            ('x3','c1') : (np.array([[7],[7],[2]], dtype=dtype), None),
            ('y1','c1') : (np.array([[6]], dtype=dtype), None),
            ('z2','c1') : (np.array([[4],[1]], dtype=dtype), None),
        }

    def tearDown(self):
        np.set_printoptions(precision=None, suppress=None)
        
    def test_mixed_subjacs(self):
        J = SparseJacobian(iteritems(self.slices))
        subjac_iter = [(k[0],k[1],subJ,idxs) for k,(subJ, idxs) in iteritems(self.subjacs)]
        J.assemble(subjac_iter, "fwd")
        self.assertEqual(J.partials.data.size, 51)
        
        check = self.subjacs.copy()
        del check['z2','a3'] # this uses idxs so won't equal the original subjac, so test separately

        for k in check:
            np.testing.assert_array_equal(J[k].A, self.subjacs[k][0])
            
        np.testing.assert_array_equal(J['z2','a3'].A, np.array([[0,0,6],[5,0,3]]))
        
        newsub = np.array([[7,7],[6,5]])
        J['z2','b2'] = newsub
        np.testing.assert_array_equal(J['z2','b2'].A, newsub)

        newsub = np.array([[7,6,5]])
        J['y1','a3'] = newsub
        np.testing.assert_array_equal(J['y1','a3'].A, newsub)
        
        J['z2','a3'] = np.array([[3,2],[9,9]])
        np.testing.assert_array_equal(J['z2','a3'].A, np.array([[2,0,3],[9,0,9]]))
        


if __name__ == '__main__':
    unittest.main()
