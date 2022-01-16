#Boolean satisfiability problem solver with continuous dynamical system
from cmath import sqrt
from mimetypes import init
import numpy as np
from abc import abstractclassmethod
from random import sample, randint, random
from scipy.integrate import odeint
from ctypes import CDLL, POINTER, c_double, c_int
#from numpy.core.defchararray import array

#from numpy.linalg.linalg import eigvalsh, solve

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

    def diff_form(self, s):
        return self.rhs(s), self.Jakobian(s)

class SAT(Problem):
    def __init__(self, cnf_file_name, n = 15, r = 4.264, so_file_name = 'cSAT.so'):
        #Initializing c functions
        self.cSAT_functions = CDLL(so_file_name)
        self.cSAT_functions.K_mi_m.restype = c_double
        self.cSAT_functions.K_m.restype = c_double
        self.cSAT_functions.gradV_i.restype = c_double
        self.c_double_p = POINTER(c_double)
        self.c_int_p = POINTER(c_int)

        # cc -fPIC -shared -o cSAT.so cSAT.c

        #Misc. init
        self.valid_solutions = None

        #Loading/generating problem
        if cnf_file_name:
            with open(cnf_file_name) as cnf_file:
                lines = cnf_file.readlines()
                super().__init__(int(lines[0].split(' ')[2]))
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
            
        else:
            literal_number = 3
            super().__init__(n) #number_of_variables
            self.clauses = []
            self.number_of_literals = []
            self.number_of_clauses = int(n*r)+1
            for i in range(self.number_of_clauses):
                clause = [elem if randint(0,1) else -elem for elem in sample(range(1, n+1), literal_number)]
                self.number_of_literals.append(literal_number)
                self.clauses.append(clause)

        self.c = np.array([[1 if (j+1) in clause else -1 if -(j+1) in clause else 0 for j in range(self.number_of_variables) ] for clause in self.clauses])

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

            #print(solution)
            for i, row in enumerate(self.clauses):
                if not check_row(row, solution):
                    incorrect_flag = True
                    break

        if incorrect_flag:
            return False
        else:
            return True

    def Jakobian_il(self,i,l,s,a):
        N_ = self.number_of_variables
        M_ = self.number_of_clauses
        summ = 0
        for m in range(M_):
            prod = 1
            for j in range(N_): 
                if j != i and j!=l:
                    prod *= (1-self.c[m,j]*s[j])
            prod *=prod
            prod *= (1-self.c[m, l]*s[l])*(3-self.c[m, l])
            summ += pow(2, 1-2*self.number_of_literals[m])*a[m]*self.c[m, i]*self.c[m, l]*prod
        return summ

    def Jakobian(self, s, a):
        N_ = self.number_of_variables
        return np.array([[self.Jakobian_il(i, l, s, a) for l in range(N_)] for i in range(N_)])

    def K_m(self, m, s):
        data1 = self.c.flatten().astype(np.int32)
        data1_p = data1.ctypes.data_as(self.c_int_p)

        data2 = s.astype(np.double)
        data2_p = data2.ctypes.data_as(self.c_double_p)
        return self.cSAT_functions.K_m(m, data2_p, data1_p, self.number_of_variables)

    def K_mi_m(self, m, i, s):
        data1 = self.c.flatten().astype(np.int32)
        data1_p = data1.ctypes.data_as(self.c_int_p)

        data2 = s.astype(np.double)
        data2_p = data2.ctypes.data_as(self.c_double_p)
        return self.cSAT_functions.K_mi_m(m, i, data2_p, data1_p, self.number_of_variables)

    def gradV_i_old(self, i, s, a):
        return sum([2*a[m]*self.c[m, i]*self.K_mi_m(m, i, s) for m in range(self.number_of_clauses) ])

    def gradV_i(self, i, s, a):
        data1 = self.c.flatten().astype(np.int32) # c
        data1_p = data1.ctypes.data_as(self.c_int_p)

        data2 = s.astype(np.double) # s
        data2_p = data2.ctypes.data_as(self.c_double_p)

        data3 = a.astype(np.double) # a
        data3_p = data3.ctypes.data_as(self.c_double_p)

        return self.cSAT_functions.gradV_i(i, data2_p, data3_p, data1_p, self.number_of_variables, self.number_of_clauses)

    def rhs(self, s, a):
        return np.array([self.gradV_i(i, s, a) for i in range(self.number_of_variables)])

    def diff_form(self, s):
        return self.rhs(s), self.Jakobian(s)

    def remove_variable(self, variable):
        new_clauses = []
        new_literals = []
        for i, clause in enumerate(self.clauses):
            new_clause = []
            for elem in clause:
                if elem > 0 and elem > variable:
                    new_elem = elem - 1
                    new_clause.append(new_elem)
                elif elem < 0 and elem < -variable:
                    new_elem = elem + 1
                    new_clause.append(new_elem)
                elif elem != variable and elem != -variable:
                    new_clause.append(elem)
            if len(new_clause) == self.number_of_literals[i]:
                new_clauses.append(new_clause)
                new_literals.append(self.number_of_literals[i])
        
        self.clauses = new_clauses
        self.number_of_variables -= 1
        self.number_of_clauses = len(new_clauses)

    def smallest_variable(self):
        used_in = [0*1 for i in range(self.number_of_variables)]
        for clause in self.clauses:
            for elem in clause:
                used_in[abs(elem)-1] += 1
        #print(used_in)
        return used_in.index(min(used_in))+1

    def write_problem_to_file(self, name):
        file_name = name + "random3SATn"+str(self.number_of_variables)+"c"+str(self.number_of_clauses)+".cnf"
        lines = []
        lines.append('p cnf ' + str(self.number_of_variables)+" "+str(self.number_of_clauses) + '\n')
        for clause in self.clauses:
            myLine = ""
            for elem in clause:
                myLine += str(elem) + " "
            lines.append(myLine + '0\n')

        with open(file_name, 'w') as mFile:
            mFile.writelines(lines)

    def get_r(self):
        return self.number_of_clauses/self.number_of_variables

    def check_solution(self, solution):
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
        for i, row in enumerate(self.clauses):
            if not check_row(row, solution):
                incorrect_flag = True
                break

        if incorrect_flag:
            return False
        else:
            return True

    def all_solutions(self):
        if self.valid_solutions is None:
            all_sols = [bin(x)[2:].rjust(self.number_of_variables, '0') for x in range(2**self.number_of_variables)]
            valid_sols = []
            for str_sol in all_sols:
                if self.check_solution([True if kar == '1' else False for kar in str_sol]):
                    valid_sols.append(str_sol)
            self.valid_solutions = valid_sols
            return valid_sols
        else:
            return self.valid_solutions

    def get_solution_index(self, solution):
        solution_str = ""
        for elem in solution:
            if elem:
                solution_str+='1'
            else:
                solution_str+='0'
        return self.all_solutions().index(solution_str)

class Lorenz(Problem):
    def __init__(self, sigma=10.0, rho=28.0, beta=2.66667):
        """The "usual" parameters for the Lorenz system are the default values, for transient chaos set rho to be in the inteval [13.93,26.06]"""
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        super().__init__(3)

    def rhs(self, s):
        return np.array([ self.sigma*( s[1]-s[0] ), s[0]*(self.rho-s[2])-s[1], s[1]*s[0]-self.beta*s[2] ])
    
    def Jakobian(self, s):
        #+s[0] miért csinálja azt amit csinál
        return np.array([ [-self.sigma, self.sigma, 0],[ self.rho-s[2], -1, -s[0] ], [s[1], s[0], -self.beta] ])

    def diff_form2(self, u):
        x,y,z = u
        f = [self.sigma * (y - x), self.rho * x - y - x * z, x * y - self.beta * z]
        Df = [[-self.sigma, self.sigma, 0], [self.rho - z, -1, -x], [y, x, -self.beta]]
        return np.array(f), np.array(Df)
    
    def diff_form(self, s):
        return self.rhs(s), self.Jakobian(s)


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

def GramSchmidt(V):
    """Gram-Schmidt orthogonalization algorithm for vectors in @param: V"""
    space_shape = np.shape(V)
    if (space_shape[0] != space_shape[1]):
        raise ValueError
        
    U = np.zeros(space_shape)
    U[0] = V[0]
    for i in range(1, space_shape[0]):
        U[i] = V[i] - sum([np.dot(U[j], V[i]) * U[j] / np.linalg.norm(U[j]) for j in range(i)])
    return U

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
        self.Ms = []
        self.Rs = []

    def save_states(self) -> None:
        """Saves the current state to member"""
        self.traj.append(self.s)
        self.times.append(self.time)
        if hasattr(self, 'M'):
            self.Ms.append(self.M)

    def step_function(self, adaptive_flag, tangental_flag, lower_diff_bound, upper_diff_bound, lower_h_bound, upper_h_bound) -> None:
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
            if (np.linalg.norm(self.diff_vec) < lower_diff_bound and not self.integrator.h > upper_h_bound):
                self.integrator.h *= 2
            elif (np.linalg.norm(self.diff_vec) > upper_diff_bound and not self.integrator.h < lower_h_bound):
                self.integrator.h /= 2
 
    def solve(self, time_limit = 2.0, adaptive_flag = True, tangental_flag = True, save_frequency = 10, lower_diff_bound = 0.00005, upper_diff_bound = 0.01, lower_h_bound = 0.00005, upper_h_bound = 0.01) -> None:
        #setting up variables
        iter_step = 0
        if not hasattr(self, 'M') and tangental_flag:
            self.M = np.identity(len(self.s))

        while time_limit >= self.time:
            self.step_function(adaptive_flag, tangental_flag, lower_diff_bound, upper_diff_bound, lower_h_bound, upper_h_bound)
            if iter_step%save_frequency:
                self.save_states()
            iter_step += 1

    def solve2(self, time_limit= 200, number_of_points = 501, upper_h_bound = 0.05):
        n = self.problem.number_of_variables
        N = n**2
        def LEC_system(u):
            #x,y,z = u[:3]
            U = u[n:n+N].reshape([n,n])
            L = u[n+N:2*n+N]
            f,Df = self.problem.diff_form(u[:n]) #the argument is the state 
            A = U.T.dot(Df.dot(U))
            dL = np.diag(A).copy()
            for i in range(n):
                A[i,i] = 0
                for j in range(i+1,n): A[i,j] = -A[j,i]
            dU = U.dot(A)
            return np.concatenate([f,dU.flatten(),dL])
        
        u0 = np.ones(n)
        U0 = np.identity(n)
        L0 = np.zeros(n)
        u0 = np.concatenate([u0, U0.flatten(), L0])
        t = np.linspace(0,time_limit,number_of_points)
        u = odeint(lambda u,t:LEC_system(u),u0,t, hmax=upper_h_bound)
        

        return t, u
        
    def solve3(self, u0, time_limit= 50, number_of_points = 501, upper_h_bound = 0.05):
        n = self.problem.number_of_variables
        N = n**2
        def LEC_system(u):
            #x,y,z = u[:3]
            U = u[n:n+N].reshape([n,n])
            L = u[n+N:2*n+N]
            f,Df = self.problem.diff_form(u[:n]) #the argument is the state 
            A = U.T.dot(Df.dot(U))
            dL = np.diag(A).copy()
            for i in range(n):
                A[i,i] = 0
                for j in range(i+1,n): A[i,j] = -A[j,i]
            dU = U.dot(A)
            return np.concatenate([f,dU.flatten(),dL])
        
        #u0 = np.ones(n)
        U0 = np.identity(n)
        L0 = np.zeros(n)
        u0 = np.concatenate([u0, U0.flatten(), L0])
        t = np.linspace(0,time_limit,number_of_points)
        u = odeint(lambda u,t:LEC_system(u),u0,t, hmax=upper_h_bound)
        
        print(self.problem.rho)
        return t, u

class CTD( Solver ):
    def __init__(self, problem, integrator, initial_s = None) -> None:
        super().__init__(problem, integrator)
        #Dynamical variables
        if initial_s is None:
            self.s = np.random.rand( self.problem.number_of_variables ) - np.array( [0.5 for i in range(self.problem.number_of_variables)] )
        else:
            self.s = initial_s
        self.a = np.array( [1 for i in range(self.problem.number_of_clauses)] )
        self.time = 0

        #Records
        self.aux = []
        self.solution = None

    def save_states(self) -> None:
        self.aux.append(self.a)
        return super().save_states()

    def step_function(self, adaptive_flag, tangental_flag=False, lower_diff_bound = 0.001, upper_diff_bound = 0.01, lower_h_bound = 0.0005, upper_h_bound = 0.01) -> None:        
        #Calculating next step
        new_s = self.integrator.step(self.s, lambda s_ : self.problem.rhs(s_, self.a))
        new_a = self.integrator.step(self.a, lambda a : np.array([a[m] * self.problem.K_m(m, self.s) for m in range(self.problem.number_of_clauses)]) )

        #Evolving tangental map
        if tangental_flag:
            Jak = self.problem.Jakobian(self.s, self.a)
            new_M = GramSchmidt( self.integrator.step(self.M, lambda M_ : np.matmul(Jak, M_)) )
            
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
            self.solution = test_solution
            return True

    def solve(self, time_limit=10.0, adaptive_flag=True, save_frequency=10, exit_type = 'ortant') -> None:
        #setting up variables
        iter_step = 0
        exit_condition = False

        while time_limit >= self.time:
            self.step_function(adaptive_flag)
            #print(self.time)
            if iter_step%save_frequency:
                self.save_states()
                if exit_type == 'ortant':
                    exit_condition = self.check_solution()
                elif exit_type == 'long_trajectory':
                    if self.check_solution():
                        N = self.problem.number_of_variables
                        sigma = 0.5
                        R = sqrt(N-1+sigma**2)
                        if np.linalg.norm(self.s) >= R:
                            exit_condition = True
                if exit_condition:
                    print("Solution found")
                    break
            iter_step += 1

    def solve2(self, time_limit=5, number_of_points = 1000, upper_h_bound = 0.01) -> None:
        n = self.problem.number_of_variables
        N = n**2
        def LEC_system(u):
            #x,y,z = u[:3]
            U = u[n:n+N].reshape([n,n])
            L = u[n+N:2*n+N]
            f,Df = self.problem.diff_form(u[:n]) #the argument is the state 
            A = U.T.dot(Df.dot(U))
            dL = np.diag(A).copy()
            for i in range(n):
                A[i,i] = 0
                for j in range(i+1,n): A[i,j] = -A[j,i]
            dU = U.dot(A)
            return np.concatenate([f,dU.flatten(),dL])
        
        u0 = np.ones(n)
        U0 = np.identity(n)
        L0 = np.zeros(n)
        u0 = np.concatenate([u0, U0.flatten(), L0])
        t = np.linspace(0,time_limit,number_of_points)
        u = odeint(lambda u,t:LEC_system(u),u0,t, hmax=upper_h_bound)
        

        return t, u

    def plot_traj(self):
        import matplotlib.pyplot as plt
        plt.grid(True)
        plt.title('Spin variables')
        for i in range(self.problem.number_of_variables):
            x = self.times
            #x = [j for j in range(len(self.traj))]
            y = [self.traj[j][i] for j in range(len(self.traj))]
            plt.plot(x, y, label=str(i+1))
        plt.legend()
        plt.show()
    
    def plot_aux(self):
        import matplotlib.pyplot as plt
        plt.grid(True)
        plt.title("Aux variables")
        for i in range(self.aux[0].size):
            #x = [j for j in range(len(self.aux))]
            x = self.times
            y = [self.aux[j][i] for j in range(len(self.aux))]
            plt.plot(x, y, label=str(i+1))
        #plt.legend()
        plt.show()

    def save_trajs(self, name):
        import matplotlib.pyplot as plt
        plt.grid(True)
        plt.title('Spin variables')
        for i in range(self.problem.number_of_variables):
            x = self.times
            y = [self.traj[j][i] for j in range(len(self.traj))]
            plt.plot(x, y, label=str(i+1))
        plt.legend()
        plt.savefig(name+".png")
        plt.clf()
        plt.close('all')

    def write_state_to_file(self, name):
        filename = name + ".csv"
        lines = []
        line = ''
        for elem in self.s:
            line += str(elem) + ','
        lines.append(line + '\n')
        line = ''
        for elem in self.a:
            line += str(elem) + ','
        lines.append(line + '\n')
        with open(filename, 'w') as mFile:
            mFile.writelines(lines)

def line_time():
    import random

    myProblem = SAT('chaotic_SAT/random3SATn11c49.cnf')
    #myProblem = SAT('chaotic_SAT/2random3SATn11c53.cnf')
    mIntegrator = RK4()
    
    number_of_solutions = len(myProblem.all_solutions())
    print('Number of solutions: {0}'.format(number_of_solutions))
    #Generate initial conditions
    initial_conditions = []
    v = np.array([1 for i in range(myProblem.number_of_variables)])
    b = np.array([random.random()/10 for i in range(myProblem.number_of_variables)])

    for i in range(500):
        t = (i-250)/300
        initial_conditions.append(v*t+b)

    dictionaries = []
    for i, init_s in enumerate(initial_conditions):
        result_dict = {}
        result_dict['t'] = []
        result_dict['init_s'] = []
        result_dict['solution'] = []
        print('Starting initial condition {0}'.format(i))
        solver = CTD(myProblem, mIntegrator, init_s)
        solver.solve()
        result_dict['t'].append( solver.time )
        result_dict['init_s'].append( init_s )
        if solver.solution:
            result_dict['solution'].append( solver.solution )
        else:
            result_dict['solution'].append( None )
        dictionaries.append(result_dict)
    
    import csv
    with open('output2.csv', 'w', newline='') as csvfile:
        fieldnames = ['t', 'init_s', 'solution']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for elem in dictionaries:
            writer.writerow(elem)

def color_area(init_s):
    myProblem = SAT('chaotic_SAT/random3SATn11c49.cnf')
    mIntegrator = RK4()
    solver = CTD(myProblem, mIntegrator, init_s)
    solver.solve()
    return solver.solution

def test_fun(init_s):
    return np.linalg.norm(init_s)

if __name__ == "__main__":
    from multiprocessing import Pool
    from copy import deepcopy

    myProblem = SAT('chaotic_SAT/random3SATn11c49.cnf')
    myProblem = SAT('chaotic_SAT/2random3SATn11c53.cnf')

    s_init_0 = [random() for i in range(myProblem.number_of_variables)]
    initial_conditions = []
    for i in range(11):
        for j in range(11):
            an_initial_condition = deepcopy(s_init_0)
            an_initial_condition[0] = i/5.01 - 0.999
            an_initial_condition[1] = j/5.01 - 0.999
            initial_conditions.append(np.array(an_initial_condition))

    output = [None for elem in initial_conditions]

    pool = Pool(processes=16)
    chunksize = int(len(initial_conditions) / pool._processes)
    for ind, res in enumerate(pool.imap_unordered(color_area, initial_conditions, chunksize)):
        output[ind] = myProblem.get_solution_index(res)
        print("Done with {0}".format(ind))

  
    with open('basin_map.out', 'w') as outputfile:
        for i in range(11):
            for j in range(11):
                outputfile.write(str(output[i*11 + j]) + ',')
            outputfile.write('\n')
