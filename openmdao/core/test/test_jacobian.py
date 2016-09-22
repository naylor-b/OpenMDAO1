import unittest

import numpy as np
from scipy.sparse import coo_matrix, issparse
from six import iteritems

from openmdao.core.jacobian import SparseJacobian, DenseJacobian

def dump(mat):
    cmat = mat.tocoo()
    for i,j,val in zip(cmat.row, cmat.col, cmat.data):
        print i,j,val
        
class TestSparseJacobian(unittest.TestCase):
    def setUp(self):
        np.set_printoptions(precision=1, suppress=True, linewidth=999)
        
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
        
        self.sparse_subjacs = {
            n : (coo_matrix(arr), idxs) for n, (arr, idxs) in iteritems(self.subjacs)
        }
        
    def _check_sparse_jacobian_fwd(self, subjacs):
        subjac_iter = [(k[0],k[1],subJ,idxs) for k,(subJ, idxs) in iteritems(subjacs)]
        J = SparseJacobian(iteritems(self.slices), subjac_iter, 'fwd')
        if issparse(subjac_iter[0][2]):
            self.assertEqual(J.partials.data.size, 42)
            convert = coo_matrix
        else:
            self.assertEqual(J.partials.data.size, 51)
            convert = lambda x: x
        
        check = subjacs.copy()
        del check['z2','a3'] # this uses idxs so won't equal the original subjac, so test separately

        for k in check:
            subjac = subjacs[k][0]
            if issparse(subjac):
                subjac = subjac.A
            np.testing.assert_array_equal(J[k].A, subjac)
            
        np.testing.assert_array_equal(J['z2','a3'].A, np.array([[0,0,6],[5,0,3]]))
        
        # NOTE: when setting a sub-jacobian into J, new array must be of the same general form as
        # the array used to build J.partials, i.e. if the subjac was sparse with 0's in certain locations,
        # the new array must also be sparse with 0's in the same locations.
        newsub = np.array([[7,0],[0,3]])
        J['z2','b2'] = convert(newsub)
        np.testing.assert_array_equal(J['z2','b2'].A, newsub)

        newsub = np.array([[9,6,0]])
        J['y1','a3'] = convert(newsub)
        np.testing.assert_array_equal(J['y1','a3'].A, newsub)
        
        J['z2','a3'] = convert(np.array([[2,0],[9,8]]))
        np.testing.assert_array_equal(J['z2','a3'].A, np.array([[0,0,2],[8,0,9]]))
        
        return J
    
    def _check_sparse_jacobian_rev(self, subjacs):
        subjac_iter = [(k[0],k[1],subJ,idxs) for k,(subJ, idxs) in iteritems(subjacs)]
        J = SparseJacobian(iteritems(self.slices), subjac_iter, 'rev')
        if issparse(subjac_iter[0][2]):
            self.assertEqual(J.partials.data.size, 42)
            convert = coo_matrix
        else:
            self.assertEqual(J.partials.data.size, 51)
            convert = lambda x: x
        
        check = subjacs.copy()
        del check['z2','a3'] # this uses idxs so won't equal the original subjac, so test separately

        for k in check:
            subjac = subjacs[k][0]
            if issparse(subjac):
                subjac = subjac.A
            np.testing.assert_array_equal(J[k].A, subjac.T)
            
        np.testing.assert_array_equal(J['z2','a3'].A, np.array([[0,0,6],[5,0,3]]).T)
        
        # NOTE: when setting a sub-jacobian into J, new array must be of the same general form as
        # the array used to build J.partials, i.e. if the subjac was sparse with 0's in certain locations,
        # the new array must also be sparse with 0's in the same locations.
        newsub = np.array([[7,0],[0,5]])
        J['z2','b2'] = convert(newsub)
        np.testing.assert_array_equal(J['z2','b2'].A, newsub.T)

        newsub = np.array([[7,6,0]])
        J['y1','a3'] = convert(newsub)
        np.testing.assert_array_equal(J['y1','a3'].A, newsub.T)
        
        J['z2','a3'] = convert(np.array([[3,0],[9,8]]))
        np.testing.assert_array_equal(J['z2','a3'].A, np.array([[0,0,3],[8,0,9]]).T)
        
        return J

    def _check_dense_jacobian_fwd(self, subjacs):
        subjac_iter = [(k[0],k[1],subJ,idxs) for k,(subJ, idxs) in iteritems(subjacs)]
        J = DenseJacobian(iteritems(self.slices), subjac_iter, "fwd")
        self.assertEqual(J.partials.size, 15*15)
        
        check = subjacs.copy()
        del check['z2','a3'] # this uses idxs so won't equal the original subjac, so test separately

        for k in check:
            subjac = subjacs[k][0]
            if issparse(subjac):
                subjac = subjac.A
            np.testing.assert_array_equal(J[k], subjac)
            
        np.testing.assert_array_equal(J['z2','a3'], np.array([[0,0,6],[5,0,3]]))
        
        newsub = np.array([[7,7],[6,5]])
        J['z2','b2'] = newsub
        np.testing.assert_array_equal(J['z2','b2'], newsub)

        newsub = np.array([[7,6,5]])
        J['y1','a3'] = newsub
        np.testing.assert_array_equal(J['y1','a3'], newsub)
        
        J['z2','a3'] = np.array([[3,2],[9,9]])
        np.testing.assert_array_equal(J['z2','a3'], np.array([[2,0,3],[9,0,9]]))
        
    def _check_dense_jacobian_rev(self, subjacs):
        subjac_iter = [(k[0],k[1],subJ,idxs) for k,(subJ, idxs) in iteritems(subjacs)]
        J = DenseJacobian(iteritems(self.slices), subjac_iter, "rev")
        self.assertEqual(J.partials.size, 15*15)
        
        check = subjacs.copy()
        del check['z2','a3'] # this uses idxs so won't equal the original subjac, so test separately

        for k in check:
            subjac = subjacs[k][0]
            if issparse(subjac):
                subjac = subjac.A
            np.testing.assert_array_equal(J[k], subjac.T)
            
        np.testing.assert_array_equal(J['z2','a3'], np.array([[0,0,6],[5,0,3]]).T)
        
        newsub = np.array([[7,7],[6,5]])
        J['z2','b2'] = newsub
        np.testing.assert_array_equal(J['z2','b2'], newsub.T)

        newsub = np.array([[7,6,5]])
        J['y1','a3'] = newsub
        np.testing.assert_array_equal(J['y1','a3'], newsub.T)
        
        J['z2','a3'] = np.array([[3,2],[9,9]])
        np.testing.assert_array_equal(J['z2','a3'], np.array([[2,0,3],[9,0,9]]).T)

    def test_sparse_dense_subjacs_fwd(self):
        J = self._check_sparse_jacobian_fwd(self.subjacs)
        
    def test_sparse_dense_subjacs_rev(self):
        J = self._check_sparse_jacobian_rev(self.subjacs)
        
    def test_dense_dense_subjacs_fwd(self):
        J = self._check_dense_jacobian_fwd(self.subjacs)
    
    def test_dense_dense_subjacs_rev(self):
        J = self._check_dense_jacobian_rev(self.subjacs)
    
    def test_sparse_sparse_subjacs_fwd(self):
        J = self._check_sparse_jacobian_fwd(self.sparse_subjacs)
        
    def test_sparse_sparse_subjacs_rev(self):
        J = self._check_sparse_jacobian_rev(self.sparse_subjacs)
        
    def test_dense_sparse_subjacs_fwd(self):
        J = self._check_dense_jacobian_fwd(self.sparse_subjacs)
    
    def test_dense_sparse_subjacs_rev(self):
        J = self._check_dense_jacobian_rev(self.sparse_subjacs)
    


if __name__ == '__main__':
    unittest.main()
