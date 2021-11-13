#Boolean satisfiability problem solver with continuous dynamical system
from typing import ValuesView
import numpy as np
from abc import abstractclassmethod
from numpy.core.defchararray import array
from numpy.linalg import linalg

from numpy.linalg.linalg import eigvalsh, solve
from scipy.linalg import orth

class Problem:
    """Abstract class representing a dynamical system"""
    def __init__(self, number_of_variables):
        self.number_of_variables = number_of_variables

    @abstractclassmethod
    def rhs(self, s):
        pass

    @abstractclassmethod
    def Jakobian(self, s):
        pass

class Rössler(Problem):
    def __init__(self, a = 0.398, b = 2.0, c=4) -> None:
        super().__init__(3)
        self.a = a
        self.b = b
        self.c = c

    def rhs(self, s):
        return np.array([-s[1]-s[2], s[0]+self.a*s[1], self.b+s[2]*(s[0]-self.c)])

    def Jakobian(self, s):
        return np.array([ [0.0, -1.0, -1.0], [1.0, self.a, 0.0], [s[2], 0.0, s[0]-self.c] ])

class SAT(Problem):
    def __init__(self, cnf_file_name) -> None:
        with open(cnf_file_name) as cnf_file:
            lines = cnf_file.readlines()
            Problem.__init__(self, int(lines[0].split(' ')[2]))

            self.number_of_clauses = int(lines[0].split(' ')[3])
            self.number_of_literals = []

            #(v)^(v)#
            clause_and = []
            for i in range(self.number_of_clauses + 1):
                literal_number = 0
                if i > 0:
                    clause_or = []
                    for variable_str in lines[i].split(' '):
                        variable = int(variable_str)
                        if variable != 0:
                            clause_or.append(variable)
                            literal_number += 1
                    clause_and.append(clause_or)
                    self.number_of_literals.append(literal_number)

            self.clauses = clause_and

        self.c = np.array([[self.c_mj(m, j) for j in range(self.number_of_variables) ] for m in range(self.number_of_clauses)])

    def verify(self, sol_file_name):

        def check_row(row, solution):
            for elem in row:
                if elem > 0:
                    #or_var = or_var or solution[elem-1]
                    if True == solution[elem-1]:
                        return True
                if elem < 0:
                    if False == solution[-elem-1]:
                        return True
                    #or_var = or_var or not solution[-elem-1]
                if elem == 0:
                    raise ValueError

        incorrect_flag = False
        with open(sol_file_name) as sol_file:
            lines = sol_file.readlines()
            solution = []
            for element_str in lines[1].split(' '):
                element = int(element_str)
                if element != 0:
                    if element > 0:
                        solution.append(True)
                    elif element < 0:
                        solution.append(False)

            print(solution)
            for i, row in enumerate(self.clauses):
                if not check_row(row, solution):
                    incorrect_flag = True
                    break

        if incorrect_flag:
            return False
        else:
            return True

    def c_mj(self, m, j):
        for variable in self.clauses[m]:
            if variable == (j+1):
                return 1
            elif variable == -(j+1):
                return -1
        return 0

    def Jakobian_il(self,i,l,s,a):
        N_ = self.number_of_variables
        M_ = self.number_of_clauses
        summ = 0
        for m in range(M_):
            prod = 1
            for j in range(N_): 
                if j != i and j!=l:
                    prod *= (1-self.c_mj(m, j)*s[j])
            prod *=prod
            prod *= (1-self.c_mj(m, l)*s[l])*(3-self.c_mj(m, l))
            summ += pow(2, 1-2*self.number_of_literals[m])*a[m]*self.c_mj(m, i)*self.c_mj(m, l)*prod
        return summ

    def Jakobian(self, s, a):
        N_ = self.number_of_variables
        return np.array([[self.Jakobian_il(i, l, s, a) for l in range(N_)] for i in range(N_)])

    def K_m(self, m, s):
        return np.prod([( 1-self.c_mj(m,j) * s[j] ) for j in range(self.number_of_variables)])

    def K_mi_m(self, m, i, s):
        K_mi = self.K_mi(m, i, s)
        return K_mi*K_mi*(1-self.c_mj(m,i)*s[i])

    def K_mi(self, m, i, s):
        return pow(2, -self.number_of_literals[m]) * np.prod([( 1-self.c_mj(m,j) * s[j] ) for j in range(self.number_of_variables) if i != j])

    def gradV_i(self, i, s, a):
        return sum([2*a[m]*self.c_mj(m, i)*self.K_mi_m(m, i, s) for m in range(self.number_of_clauses) ])

    def rhs(self, s, a):
        return np.array([self.gradV_i(i, s, a) for i in range(self.number_of_variables)])

class Lorenz(Problem):
    def __init__(self, sigma=10.0, rho=28.0, beta=2.66667):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        super().__init__(3)

    def rhs(self, s):
        return np.array([ self.sigma*( s[1]-s[0] ), s[0]*(self.rho-s[2])-s[1], s[1]*s[0]-self.beta*s[2] ])
    
    def Jakobian(self, s):
        return np.array([ [-self.sigma, self.sigma, 0],[ self.rho-s[2], -1, -s[0] ], [s[1], s[0], -self.beta] ])

#Numerical integrators

class Integrator:
    def __init__(self, Nmax = 10000, h = 0.0005) -> None:
        self.h = h
        self.Nmax = Nmax

class RK5( Integrator ):
    def __init__(self, Nmax = 10000, h = 0.0025) -> None:
        Integrator.__init__(self, Nmax, h)

    def step(self, y, f, df):
        k1 = self.h * f(y)
        k2 = self.h * f(y + k1*1/3 + self.h * np.dot(df(y), k1))
        k3 = self.h * f(y + k1*152/125 + k2*252/125 - self.h * 44/125 * np.dot(df(y), k1))
        k4 = self.h * f(y + k1*19/2 - k2*72/7 + k3*25/14 + self.h * 5/2 * np.dot(df(y), k1))
        return y + 5/48*k1 + 27/56*k2 + 125/336*k3 + 1/24*k4

class RK4( Integrator ):
    def __init__(self, y_init = None, Nmax = 10, h = 0.0001) -> None:
        Integrator.__init__(self, Nmax, h)

    def step(self, y, f):
        k1 = self.h*f(y)
        k2 = self.h*f(y+0.5*k1)
        k3 = self.h*f(y+0.5*k2)
        k4 = self.h*f(y+k3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0 

class ForwardEuler( Integrator ):
    """Explicit forward euler method integrator"""
    def __init__(self, Nmax = 10000, h = 0.0025) -> None:
        Integrator.__init__(self, Nmax, h)
    
    def step(self, y, f):
        return y + self.h*f(y)

#Numerical solvers

class Solver:
    def __init__(self, problem, integrator) -> None:
        """Constructor of abstract solver class"""
        self.problem = problem
        self.integrator = integrator

        #Dynamical variables
        self.time = 0.0
        self.s = np.array( [0.0 for i in range(self.problem.number_of_variables)] )
        self.M = np.eye(self.problem.number_of_variables, dtype=np.longdouble)

        #Records
        self.traj = []
        self.diff_vec = None
        self.times = []
        self.axis_lengths = []

    def save_states(self) -> None:
        """Saves the current state to member"""
        self.traj.append(self.s.copy())
        self.times.append(self.time)

        logr = []
        U, s, rotation = np.linalg.svd(np.array(self.M, dtype=np.float32))
        radii = 1.0/np.sqrt(s)
        #for i, eig_val_i in enumerate(eig_vals):
            #logri = np.log(abs(1/eig_val_i**2)) + sum([np.log( R[i] ) for R in self.axis_lengths])
            #logr.append(logri)
        #self.logr_ij.append(logr)
        self.axis_lengths.append(radii)
        

    def step_function(self, adaptive_flag, lower_diff_bound, upper_diff_bound, lower_h_bound, upper_h_bound) -> None:
        valid_step = True
        #Calculating next step
        new_s = self.integrator.step(self.s, lambda s_ : self.problem.rhs(s_))
        new_M = (np.eye(self.problem.number_of_variables) @ self.problem.Jakobian(self.s)) @ self.M * self.integrator.h

        #Adaptive step size
        if adaptive_flag:
            self.diff_vec = self.s- new_s
            if (np.linalg.norm(self.diff_vec) < lower_diff_bound and not self.integrator.h > upper_h_bound):
                self.integrator.h *= 2
            elif (np.linalg.norm(self.diff_vec) > upper_diff_bound and not self.integrator.h < lower_h_bound):
                self.integrator.h /= 2
                valid_step=False

        if valid_step:
            #Updating dynamic variables
            self.time += self.integrator.h
            self.s = new_s
            self.M = new_M
 
    def solve(self, time_limit = 2.0, adaptive_flag = True, save_frequency = 2, lower_diff_bound = 0.00001, upper_diff_bound = 0.001, lower_h_bound = 0.00005, upper_h_bound = 0.001) -> None:
        #setting up variables
        iter_step = 0
        while time_limit >= self.time:
            self.step_function(adaptive_flag, lower_diff_bound, upper_diff_bound, lower_h_bound, upper_h_bound)
            if iter_step%save_frequency or save_frequency == 1:
                self.M = self.orthogonalize(self.M)
                self.save_states()

                #Numerical stabilty
                renormalization_needed = False
                for radius in self.axis_lengths[-1]:
                    if radius > 10e15 or 10e-15 > radius:
                        renormalization_needed = True
                if renormalization_needed:
                    self.renormalize()
            iter_step += 1

    def orthogonalize(self, M):
        """Gram-Schidt procedure"""
        V = np.zeros(np.shape(M), dtype=np.longdouble)
        for i, u_i in enumerate(M):
            if i == 0:
                V[i] = u_i
            else:
                ms = np.zeros(np.shape(V[0]))
                for j in range(i):
                    proj = np.dot(u_i, V[j])
                    scale = 1/np.linalg.norm(V[j])**2
                    ms += V[j]*proj*scale
                V[i] = u_i - ms
        return V

    def renormalize(self) -> None:
        print("Renormalized at" + str(self.time))
        for elem in self.M:
            elem /= np.linalg.norm(elem)


class CTD( Solver ):
    def __init__(self, problem, integrator) -> None:
        super().__init__(problem, integrator)
        #Dynamical variables
        #self.s = np.random.rand( self.SATproblem.number_of_variables ) - np.array( [0.5 for i in range(self.SATproblem.number_of_variables)] )
        self.a = np.array( [1 for i in range(self.problem.number_of_clauses)] )
        self.time = 0

        #Records
        self.aux = []

    def save_states(self) -> None:
        self.aux.append(self.a)
        return super().save_states()

    def step_function(self, adaptive_flag, tangental_flag, lower_diff_bound = 0.00005, upper_diff_bound = 0.01, lower_h_bound = 0.00005, upper_h_bound = 0.01) -> None:        
        #Calculating next step
        new_s = self.integrator.step(self.s, lambda s_ : self.problem.rhs(s_, self.a))
        new_a = self.integrator.step(self.a, lambda a : np.array([a[m] * self.problem.K_m(m, self.s) for m in range(self.problem.number_of_clauses)]) )

        #Evolving tangental map
        if tangental_flag:
            Jak = self.problem.Jakobian(self.s, self.a)
            new_M = self.integrator.step(self.M, lambda M_ : np.matmul(Jak, M_) )
            
        #Adaptive step size
        if adaptive_flag:
            diff_vec_s = self.s- new_s
            diff_vec_a = self.a-new_a
            if tangental_flag:
                diff_vec_M = (self.M-new_M).flatten()
                self.diff_vec = np.concatenate((diff_vec_s, diff_vec_a, diff_vec_M), axis=None)
            else:
                self.diff_vec = np.concatenate((diff_vec_s, diff_vec_a), axis=None)
            if (np.linalg.norm(self.diff_vec) < lower_diff_bound and not self.integrator.h > upper_h_bound):
                self.integrator.h *= 2
            elif (np.linalg.norm(self.diff_vec) > upper_diff_bound and not self.integrator.h < lower_h_bound):
                self.integrator.h /= 2

        #Updating dynamic variables
        self.time += self.integrator.h
        self.s = new_s
        self.a = new_a
        if tangental_flag:
            self.M = new_M

    def check_solution(self, sol = None) -> bool:
        def check_row(row, solution):
            for elem in row:
                if elem > 0:
                    #or_var = or_var or solution[elem-1]
                    if True == solution[elem-1]:
                        return True
                if elem < 0:
                    if False == solution[-elem-1]:
                        return True
                    #or_var = or_var or not solution[-elem-1]
                if elem == 0:
                    raise ValueError

        incorrect_flag = False
        if not sol:
            test_solution = [True if elem > 0 else False for elem in self.s]
        else:
            test_solution = sol
        for i, row in enumerate(self.problem.clauses):
            if not check_row(row, test_solution):
                incorrect_flag = True
                break

        if incorrect_flag:
            return False
        else:
            return True

    def solve(self, time_limit=2, adaptive_flag=True, tangental_flag=True, save_frequency=10) -> None:
        #setting up variables
        iter_step = 0
        if not hasattr(self, 'M') and tangental_flag:
            self.M = np.identity(len(self.s))

        while time_limit >= self.time:
            self.step_function(adaptive_flag, tangental_flag)
            if iter_step%save_frequency:
                self.save_states()
                if self.check_solution():
                    print("Solution found")
                    break
            iter_step += 1

def plot_ellispe(matrix):
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    A = np.array(matrix, dtype=np.float32)
    center = [0.0, 0.0, 0.0]

    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(A)
    radii = 1.0/np.sqrt(s)

    # calculate cartesian coordinates for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 60)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z,  rstride=3, cstride=3, linewidth=0.1, alpha=1, shade=True)
    plt.show()