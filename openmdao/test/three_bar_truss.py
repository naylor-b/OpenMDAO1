"""3 Bar Truss Problem.

- 3 continuous design variables -  Cross sectional area of each bar.

- 3 discrete material choices for each bar. Complete enumeration requires 64 continuous optimization.

"""

import numpy as np

from openmdao.core.component import Component


def stress_calc(A, E):
    """ Calculate stress in a 3-bar truss given area and Young's modulus."""
    
    P  = 120000.0
    L = np.array([np.sqrt(1.2**2 + 1.2**2), 1.2, np.sqrt(1.2**2 + 1.2**2)])

    #Local stiffness matrix
    theta1 = -45.0*np.pi/180.0
    theta2 = -90.0*np.pi/180.0
    theta3 = -135*np.pi/180.0

    K0 = (E[0]*A[0]/L[0])*np.dot(np.array([[np.cos(theta1), np.sin(theta1)]]).T,
                                          np.array([[np.cos(theta1), np.sin(theta1)]]))
    K1 = (E[1]*A[1]/L[1])*np.dot(np.array([[np.cos(theta2), np.sin(theta2)]]).T,
                                          np.array([[np.cos(theta2), np.sin(theta2)]]))
    K2 = (E[2]*A[2]/L[2])*np.dot(np.array([[np.cos(theta3), np.sin(theta3)]]).T,
                                          np.array([[np.cos(theta3), np.sin(theta3)]]))

    # Global (total) stiffness matrix
    K = K0 + K1 + K2

    # Load vector
    theta4 = -65.0*np.pi/180.0
    p = P*np.array([[np.cos(theta4), np.sin(theta4)]]).T
    # Displacement matrix
    u = np.dot(np.linalg.inv(K), p)

    #Delta change in length
    DL = np.zeros([3])
    DL[0] = np.sqrt((-L[0]*np.cos(theta1) - u[0])**2 + (-L[0]*np.sin(theta1) - u[1])**2) - L[0]
    DL[1] = np.sqrt((-L[1]*np.cos(theta2) - u[0])**2 + (-L[1]*np.sin(theta2) - u[1])**2) - L[1]
    DL[2] = np.sqrt((-L[2]*np.cos(theta3) - u[0])**2 + (-L[2]*np.sin(theta3) - u[1])**2) - L[2]

    #Stress in each element
    sigma = E*DL/L
    return sigma


class ThreeBarTruss(Component):
    """ 3 Bar truss problem with 3 continuous design variables and 3 discrete
    material choices. Material chosen as follows:
    
            1 Aluminum
            2 Titanium
            3 Steel
            4 Nickel
            
    This component calculates the total mass of the truss.
    """

    def __init__(self):
        super(ThreeBarTruss, self).__init__()

        # Continuous Inputs
        self.add_param('area1', 0.0, units='cm**2',
                       desc='Cross-sectional area of beam 1')
        self.add_param('area2', 0.0, units='cm**2',
                       desc='Cross-sectional area of beam 2')
        self.add_param('area3', 0.0, units='cm**2',
                       desc='Cross-sectional area of beam 3')

        # Discrete Inputs
        self.add_param('mat1', 1, lower=1, upper=4,
                       desc='Material ID of beam 1')
        self.add_param('mat2', 1, lower=1, upper=4,
                       desc='Material ID of beam 2')
        self.add_param('mat3', 1, lower=1, upper=4,
                       desc='Material ID of beam 3')

        # Outputs
        self.add_output('mass', val=0.0)
        self.add_output('stress', val=np.zeros((3, 1)))
        
        self.rho = { 1 : 2700.0,
                     2 : 4500.0, 
                     3 : 7872.0,
                     4 : 8800.0 }
        
        self.E = { 1 : 68.9e9,
                   2 : 116.0e9, 
                   3 : 205.0e9,
                   4 : 207.0e9 }
        
        self.sigma_y = { 1 : 55.2e6,
                         2 : 140.0e6, 
                         3 : 285.0e6,
                         4 : 59.0e6 }        
        
        self.deriv_options['type'] = 'fd'
        
    def solve_nonlinear(self, params, unknowns, resids):
        """ Define the function f(xI, xC)
        Here xI is integer and xC is continuous"""

        area1 = params['area1']
        area2 = params['area2']
        area3 = params['area3']
        mat1 = int(params['mat1'])
        mat2 = int(params['mat2'])
        mat3 = int(params['mat3'])
        
        # Convert to meters^2 from cm^2
        area1 /= 1.0e4
        area2 /= 1.0e4
        area3 /= 1.0e4
        
        len1 = np.sqrt(1.2**2 + 1.2**2)
        len2 = 1.2
        len3 = np.sqrt(1.2**2 + 1.2**2)
        
        rho1 = self.rho[mat1]
        rho2 = self.rho[mat2]
        rho3 = self.rho[mat3]
        
        unknowns['mass'] = rho1*area1*len1 + rho2*area2*len2 + rho3*area3*len3

        E1 = self.E[mat1]
        E2 = self.E[mat2]
        E3 = self.E[mat3]
        sigma_y1 = self.sigma_y[mat1]
        sigma_y2 = self.sigma_y[mat2]
        sigma_y3 = self.sigma_y[mat3]
        
        E = np.array([E1, E2, E3])
        A = np.array([area1, area2, area3])
        sigma_y = np.array([sigma_y1, sigma_y2, sigma_y3])
            
        sigma = stress_calc(A, E)
        unknowns['stress'] = (np.abs(sigma)/sigma_y)


class ThreeBarTrussVector(Component):
    """ 3 Bar truss problem with 3 continuous design variables and 3 discrete
    material choices. Material chosen as follows:
    
            1 Aluminum
            2 Titanium
            3 Steel
            4 Nickel
            
    This component calculates the total mass of the truss.
    
    This version places all areas in a single vector and all materials in
    a single vector for test purporses.
    """

    def __init__(self):
        super(ThreeBarTrussVector, self).__init__()

        # Continuous Inputs
        self.add_param('area', np.array([0.0, 0.0, 0.0]), units='cm**2',
                       desc='Cross-sectional area of beams')

        # Discrete Inputs
        self.add_param('mat', np.array([1, 1, 1]), lower=1, upper=4,
                       desc='Material ID of beams')

        # Outputs
        self.add_output('mass', val=0.0)
        self.add_output('stress', val=np.zeros((3, 1)))
        
        self.rho = { 1 : 2700.0,
                     2 : 4500.0, 
                     3 : 7872.0,
                     4 : 8800.0 }
        
        self.E = { 1 : 68.9e9,
                   2 : 116.0e9, 
                   3 : 205.0e9,
                   4 : 207.0e9 }
        
        self.sigma_y = { 1 : 55.2e6,
                         2 : 140.0e6, 
                         3 : 285.0e6,
                         4 : 59.0e6 }        
        
        self.deriv_options['type'] = 'fd'
        
    def solve_nonlinear(self, params, unknowns, resids):
        """ Define the function f(xI, xC)
        Here xI is integer and xC is continuous"""

        area = params['area']
        mat = int(params['mat'])
        
        # Convert to meters^2 from cm^2
        area /= 1.0e4
        
        length = np.array(np.sqrt(1.2**2 + 1.2**2),
                          1.2,
                          np.sqrt(1.2**2 + 1.2**2))
        
        rho = np.zeros((3))
        rho[0] = self.rho[mat1]
        rho[1] = self.rho[mat2]
        rho[2] = self.rho[mat3]
        
        unknowns['mass'] = rho*area*length

        E = np.zeros((3))
        E[0] = self.E[mat1]
        E[1] = self.E[mat2]
        E[2] = self.E[mat3]
        sigma_y = np.zeros((3))
        sigma_y[0] = self.sigma_y[mat1]
        sigma_y[1] = self.sigma_y[mat2]
        sigma_y[2] = self.sigma_y[mat3]
        
        sigma = stress_calc(A, E)
        unknowns['stress'] = (np.abs(sigma)/sigma_y)
