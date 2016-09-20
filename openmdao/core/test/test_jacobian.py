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
            ('a3','x3') : (np.array([[1,0,6],[5,5,9],[0,0,3]], dtype=dtype), None),
            ('a3','y1') : (np.array([[1],[7],[0]], dtype=dtype), None),
            ('a3','z2') : (np.array([[6,0],[3,5]], dtype=dtype), [2,0]),
            ('b2','b2') : (np.array([[0,8],[9,7]], dtype=dtype), None),
            ('b2','x3') : (np.array([[9,9,0],[8,0,8]], dtype=dtype), None),
            ('b2','y1') : (np.array([[9],[3]], dtype=dtype), None),
            ('b2','z2') : (np.array([[5,0],[0,8]], dtype=dtype), None),
            ('c1','x3') : (np.array([[7,7,2]], dtype=dtype), None),
            ('c1','y1') : (np.array([[6]], dtype=dtype), None),
            ('c1','z2') : (np.array([[4,1]], dtype=dtype), None),
        }

    def tearDown(self):
        np.set_printoptions(precision=None, suppress=None)
        
    def test_mixed_subjacs(self):
        J = SparseJacobian(iteritems(self.slices))
        subjac_iter = [(k[0],k[1],subJ,idxs) for k,(subJ, idxs) in iteritems(self.subjacs)]
        J.assemble(subjac_iter, "fwd")
        self.assertEqual(J.partials.data.size, 51)
        
        check = self.subjacs.copy()
        del check['a3','z2'] # this uses idxs so won't equal the original subjac

        for k in check:
            np.testing.assert_array_equal(J[k].A, self.subjacs[k][0])
            
        np.testing.assert_array_equal(J['a3','z2'].A, np.array([[3,5],[0,0],[6,0]]))
        
        newsub = np.array([[7,7],[6,5]])
        J['b2','z2'] = newsub
        np.testing.assert_array_equal(J['b2','z2'].A, newsub)

        newsub = np.array([[7],[6],[5]])
        J['a3','y1'] = newsub
        np.testing.assert_array_equal(J['a3','y1'].A, newsub)
        
        J['a3','z2'] = np.array([[3,2],[9,9]])
        np.testing.assert_array_equal(J['a3','z2'].A, np.array([[9,9],[0,0],[3,2]]))
        


if __name__ == '__main__':
    unittest.main()
