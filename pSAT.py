#Boolean satisfiability problem solver with continuous dynamical system
import numpy as np
from abc import abstractclassmethod
from numpy.core.defchararray import array

from numpy.linalg.linalg import eigvalsh, solve

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
    def __init__(self, a = 0.398, b = 2.0, c=4):
        super().__init__(3)
        self.a = a
        self.b = b
        self.c = c

    def rhs(self, s):
        return np.array([-s[1]-s[2], s[0]+self.a*s[1], self.b+s[2]*(s[0]-self.c)])

    def Jakobian(self, s):
        return np.array([ [0.0, -1.0, -1.0], [1.0, self.a, 0.0], [s[2], 0.0, s[0]-self.c] ])

class SAT(Problem):
    def __init__(self, cnf_file_name):
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
                    #print(row)
                    #print(i)
                    #print([solution[abs(x)] for x in row])
                    #raise RuntimeError
                    break

        if incorrect_flag:
            return False
            #print('The solution "'+ sol_file_name +'" to '+ assignment +' is NOT correct')
        else:
            return True
            #print('The solution "'+ sol_file_name +'" to '+ assignment +' is correct')

    def c_mj(self, m, j):
        for variable in self.clauses[m]:
            if variable == (j+1):
                return 1
            elif variable == -(j+1):
                return -1
        return 0

#Numerical integrators

class Integrator:
    def __init__(self, Nmax = 10000, h = 0.0025) -> None:
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
    def __init__(self, y_init = None, Nmax = 10, h = 0.0025) -> None:
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

        #Records
        self.traj = []
        self.diff_vec = None
        self.times = []

    def save_states(self):
        """Saves the current state to member"""
        self.traj.append(self.s)
        self.times.append(self.time)

    def step_function(self, adaptive_flag, tangental_flag):
        #Calculating next step
        new_s = self.integrator.step(self.s, lambda s_ : self.problem.rhs(s_))

        #Evolving tangental map
        if tangental_flag:
            Jak = self.problem.Jakobian(self.s)
            new_M = self.integrator.step(self.M, lambda M_ : np.matmul(Jak, M_) )
            self.M = new_M

        #Updating dynamic variables
        self.time += self.integrator.h
        self.s = new_s

        #Adaptive step size
        if adaptive_flag:
            self.diff_vec = self.s- new_s
            if (np.linalg.norm(self.diff_vec) < 0.00005 and not self.integrator.h > 0.2):
                self.integrator.h *= 2
            elif (np.linalg.norm(self.diff_vec) > 0.01 and not self.integrator.h < 0.00005):
                self.integrator.h /= 2
 
    def solve(self, time_limit = 2.0, adaptive_flag = True, tangental_flag = True, save_frequency = 10):
        #setting up variables
        iter_step = 0
        if not hasattr(self, 'M') and tangental_flag:
            self.M = np.identity(len(self.s))

        while time_limit >= self.time:
            self.step_function(adaptive_flag, tangental_flag)
            if iter_step%save_frequency:
                self.save_states()
            iter_step += 1

    def plot_traj(self):
        import matplotlib.pyplot as plt
        plt.grid(True)
        plt.title('Dynamical variables')
        for i in range(self.problem.number_of_variables):
            x = self.times
            y = [self.traj[j][i] for j in range(len(self.traj))]
            plt.plot(x, y, label=str(i+1))
        plt.legend()
        plt.show()

    def get_Lyapunov_exponents(self):
        if self.M is None:
            return None
        else:
            eigvals = np.linalg.eigvalsh( np.dot(self.M, np.transpose(self.M)) )
            return np.log(eigvals)/(2*self.time)

    def get_Lyapunov_convergence(self, max_time = 100, step_size = 0.25):
        result = []
        for i in range(int(max_time/step_size)):
            time = step_size*i
            self.solve(time)
            result.append( (self.get_Lyapunov_exponents(), time) )
        return result
