#########################################
#                                       #
# Boolean satisfiability problem solver #
#  with continuous dynamical system     #
#                                       #
# Written by Áron Vízkeleti             #
#       on 2021-10-10                   #
#       last modified 2022-04-04        #
#                                       #
#########################################

from cmath import nan
from math import sqrt, pi, sin
from xmlrpc.client import Boolean
import numpy as np
from abc import abstractclassmethod
from random import sample, randint, random
from scipy.integrate import solve_ivp
from ctypes import CDLL, POINTER, c_double, c_int

#Constants

ORTANT = 0
CONVERGENCE_RADIUS = -1
NEGATIVE_AUX = -2
RHS_TYPE_ONE = 1
RHS_TYPE_TWO = 2
RHS_TYPE_THREE = 3
RHS_TYPE_FOUR = 4
RHS_TYPE_FIVE = 5

#Numerical integrator(s)

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
        k2 = self.h*f(y+0.5*k1) #f(arg2)
        k3 = self.h*f(y+0.5*k2) #f(arg3)
        k4 = self.h*f(y+k3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0 

class ForwardEuler( Integrator ):
    """Explicit forward euler method integrator"""
    def __init__(self, Nmax = 10000, h = 0.0025) -> None:
        Integrator.__init__(self, Nmax, h)
    
    def step(self, y, f):
        return y + self.h*f(y)

#Problem definition(s)

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
    """Python class representing the 3D Rössler system"""
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
        return np.array([ [-self.sigma, self.sigma, 0],[ self.rho-s[2], -1, -s[0] ], [s[1], s[0], -self.beta] ])
    
    def diff_form(self, s):
        return self.rhs(s), self.Jakobian(s)

class SAT(Problem):
    """Class representation of the continuous dynamical system version of a boolean satisfiability ptoblem"""
    def __init__(self, cnf_file_name, so_file_name, n = 15, alpha = 4.264, literal_number = 3, rhs_type = RHS_TYPE_ONE):
        """
        Constructor
        @param cnf_file_name: cnf-file defining the problem, if set to None, generates a random problem
        @param so_file_name: c/c++ library containing (hopefully fast) implementation of rhs and jakobian matrices
        @param n: optional, number of variables in randomly generated problem
        @param alpha: optional, ration of clauses (w.r.t n) in randomly generated problem
        @param literal_number: optional, defines the length of clauses (default is 3)
        @param rhs_type: optional, selects type of rhs (RHS_TYPE_ONE = 1) (RHS_TYPE_TWO = 2)
        """
        #Misc. init
        self.valid_solutions = None
        self.rhs_type = rhs_type
        self.alpha = None
        

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

        #Randomly generating a sat problem
        else:
            super().__init__(n) #number_of_variables
            self.clauses = []
            self.number_of_literals = []
            self.number_of_clauses = int(n*alpha)+1
            for i in range(self.number_of_clauses):
                clause = [elem if randint(0,1) else -elem for elem in sample(range(1, n+1), literal_number)]
                self.number_of_literals.append(literal_number)
                self.clauses.append(clause)

        #Generating the clause matrix
        self.c = np.array([[1 if (j+1) in clause else -1 if -(j+1) in clause else 0 for j in range(self.number_of_variables) ] for clause in self.clauses])
        self.alpha = self.get_alpha()

        #Loading c_functions
        if not so_file_name:
            self.cSAT_functions = None
        else:
            self.cSAT_functions = CDLL(so_file_name)
            self.cSAT_functions.rhs1.restype = None
            self.cSAT_functions.rhs2.restype = None
            self.cSAT_functions.rhs3.restype = None
            self.cSAT_functions.rhs4.restype = None
            self.cSAT_functions.rhs5.restype = None
            self.cSAT_functions.jacobian1.restype = None
            self.cSAT_functions.jacobian2.restype = None

    def Jakobian(self, y):
        """Jakobian matrix of the CTDS"""
        N_ = self.number_of_variables
        M_ = self.number_of_clauses
        
        if not self.cSAT_functions:
            s = y[:N_]
            a = y[N_:M_]
            if self.rhs_type == RHS_TYPE_ONE:
                raise NotImplementedError
            elif self.rhs_type == RHS_TYPE_TWO:
                return np.array([[self.Jakobian_il(i, l, s, a) for l in range(N_)] for i in range(N_)])
        else:
            clause_matrix = self.c.flatten().astype(np.int32) # c
            clause_matrix_pointer = clause_matrix.ctypes.data_as(POINTER(c_int))

            state = y.astype(np.double) # s & a
            state_pointer = state.ctypes.data_as(POINTER(c_double))

            result = np.empty((self.number_of_variables + self.number_of_clauses)**2)
            result = result.astype(np.double) # (s + a)**2
            result_pointer = result.ctypes.data_as(POINTER(c_double))
        
            if self.rhs_type == RHS_TYPE_ONE:
                self.cSAT_functions.jacobian1(self.number_of_variables, self.number_of_clauses, clause_matrix_pointer, state_pointer, result_pointer)
            elif self.rhs_type == RHS_TYPE_TWO:
                raise NotImplementedError
                #self.cSAT_functions.jacobian2(self.number_of_variables, self.number_of_clauses, clause_matrix_pointer, state_pointer, result_pointer)
            return result.reshape([N_+M_, N_+M_])

    def rhs(self, t, y):
        """Right-hand side of the differential equation defining the system"""
        N_ = self.number_of_variables
        if not self.cSAT_functions: #This condition should be moved outside of solver
            s = y[:N_]
            a = y[N_:]
            if self.rhs_type == RHS_TYPE_ONE:
                ds = np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) for i in range(self.number_of_variables) ])
                da = np.array([a[m]*self.K(m, s) for m in range(self.number_of_clauses)])
                return np.concatenate((ds, da), axis=None)
            elif self.rhs_type == RHS_TYPE_TWO:
                ds = np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) for i in range(self.number_of_variables) ])
                da = np.array([a[m]*(self.K(m, s)**2) for m in range(self.number_of_clauses)])
                return np.concatenate((ds, da), axis=None)
            elif self.rhs_type == RHS_TYPE_THREE:
                b = 0.0725
                a_ = sum(a)/self.number_of_clauses
                constant = 0.5*pi*b*self.alpha * a_
                ds = np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) + constant*sin(pi*s[i])  for i in range(self.number_of_variables) ])
                da = np.array([a[m]*(self.K(m, s)**2) for m in range(self.number_of_clauses)])
                return np.concatenate((ds, da), axis=None)
            elif self.rhs_type == RHS_TYPE_FOUR or self.rhs_type == RHS_TYPE_FIVE:
                b = 0.0725
                a_ = sum(a)/len(a)
                constant = 0.5*pi*b*self.alpha * a_
                ds = np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) + constant*sin(pi*s[i])  for i in range(self.number_of_variables) ])
                da = np.array([a[m]*(self.K(m, s)) for m in range(self.number_of_clauses)])
                return np.concatenate((ds, da), axis=None)
            elif self.rhs_type == RHS_TYPE_FIVE:
                b = 0.0725
                a_ = sum(a)/len(a)
                constant = 0.5*pi*b*self.alpha * a_
                ds = (-1)*np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) + constant*sin(pi*s[i])  for i in range(self.number_of_variables) ])
                da = (-1)*np.array([a[m]*(self.K(m, s)) for m in range(self.number_of_clauses)])
                return np.concatenate((ds, da), axis=None)
        else:
            clause_matrix = self.c.flatten().astype(np.int32) # c
            clause_matrix_pointer = clause_matrix.ctypes.data_as(POINTER(c_int))

            state = y.astype(np.double) # s & a
            state_pointer = state.ctypes.data_as(POINTER(c_double))

            result = np.empty(self.number_of_variables + self.number_of_clauses)
            result = result.astype(np.double) # s & a
            result_pointer = result.ctypes.data_as(POINTER(c_double))
            
            if self.rhs_type == RHS_TYPE_ONE:
                self.cSAT_functions.rhs1(self.number_of_variables, self.number_of_clauses, clause_matrix_pointer, state_pointer, result_pointer)
            elif self.rhs_type == RHS_TYPE_TWO:
                self.cSAT_functions.rhs2(self.number_of_variables, self.number_of_clauses, clause_matrix_pointer, state_pointer, result_pointer)
            elif self.rhs_type == RHS_TYPE_THREE:
                self.cSAT_functions.rhs3(self.number_of_variables, self.number_of_clauses, clause_matrix_pointer, state_pointer, result_pointer)
            elif self.rhs_type == RHS_TYPE_FOUR:
                self.cSAT_functions.rhs4(self.number_of_variables, self.number_of_clauses, clause_matrix_pointer, state_pointer, result_pointer)
            elif self.rhs_type == RHS_TYPE_FIVE:
                self.cSAT_functions.rhs5(self.number_of_variables, self.number_of_clauses, clause_matrix_pointer, state_pointer, result_pointer)
            return result
    
    def K(self, m, s):
        """Clause term, as defined in nature physics letter doi:10.1038/NPHY2105"""
        return pow(2, -self.number_of_literals[m])*np.prod([( 1-self.c[m,j] * s[j] ) for j in range(self.number_of_variables)])

    def k(self, m, i, s):
        """Modified clause term, as defined in nature physics letter doi:10.1038/NPHY2105"""
        return pow(2, -self.number_of_literals[m])*np.prod([( 1-self.c[m,j] * s[j] ) for j in range(self.number_of_variables) if i != j])

    def remove_variable(self, variable):
        """
        Removes a variable and all the clauses the variable appeared in.
        @param variable: index of the variable to be removed
        """
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
        self.number_of_literals = new_literals

    def smallest_variable(self):
        """Returns the index of the varibale that appears in the smallest number of clauses"""
        used_in = [0*1 for i in range(self.number_of_variables)]
        for clause in self.clauses:
            for elem in clause:
                used_in[abs(elem)-1] += 1
        #print(used_in)
        return used_in.index(min(used_in))+1

    def write_problem_to_file(self, name):
        """Generates cnf file of the problem"""
        file_name = name + ".cnf"
        lines = []
        lines.append('p cnf ' + str(self.number_of_variables)+" "+str(self.number_of_clauses) + '\n')
        for clause in self.clauses:
            myLine = ""
            for elem in clause:
                myLine += str(elem) + " "
            lines.append(myLine + '0\n')

        with open(file_name, 'w') as mFile:
            mFile.writelines(lines)

    def get_alpha(self):
        if not self.alpha:
            self.alpha = self.number_of_clauses/self.number_of_variables
        return self.alpha

    def check_solution(self, solution) -> Boolean:
        """
        Returns true (or false) if the given solution satisfies (or does not) the SAT problem
        @param solution: a list of boolean values
        """
        if len(solution) != self.number_of_variables:
            raise ValueError

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
            return False #Solution does not solve the sat problem
        else:
            #self.solution = "".join(['1' if elem else '0' for elem in test_solution])
            return True #Solutions solves the problem

    def all_solutions(self):
        """Returns a list of all solutions in a list. This uses gready algorithm, do not use for big problems"""
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
        """Returns the index of a solution given in a binary string (zeroes and ones as a string)"""
        return self.all_solutions().index( solution )

    def Hamming_distance(self, sol1, sol2):
        if len(sol1) != len(sol2):
            raise ValueError
        else:
            distance = 0
            for b1, b2 in zip(sol1, sol2):
                if int(b1) ^ int(b2):
                    distance += 1
            return distance

    def get_clusters(self):
        """Generates dictionary of solution clusters, only use for small problems"""
        def in_dict_list(elem, dict):
            for val in dict.values():
                if elem in val:
                    return True
            return False
        
        def get_sol_key(solution, clusters):
            for key, val in clusters.items():
                for sol in val:
                    if sol == solution:
                        return key
            
            return None

        solutions = self.all_solutions()
        clusters = {}

        for cluster_idx, sol in enumerate(solutions):
            if not in_dict_list(sol, clusters):
                clusters[cluster_idx] = [sol]
            for sol2 in solutions:
                if sol2 != sol and not in_dict_list(sol2, clusters) and self.Hamming_distance(sol, sol2) == 1:
                        clusters[get_sol_key(sol, clusters)].append(sol2)
        
        return clusters

#Numerical solver definition(s)

class CTD:
    def __init__(self, problem, integrator = None, initial_s = None, random_aux = False) -> None:
        self.problem = problem
        self.state = np.empty(problem.number_of_variables + problem.number_of_clauses)

        #Dynamical variables
        if initial_s is None:
            self.state[0:problem.number_of_variables] = np.array([2*random() -1 for i in range(problem.number_of_variables)])
        else:
            self.state[0:problem.number_of_variables] = initial_s
        if random_aux == True:
            self.state[problem.number_of_variables:] = np.array( [random()*15 for i in range(self.problem.number_of_clauses)] )
        else:
            self.state[problem.number_of_variables:] = np.ones(self.problem.number_of_clauses)
        self.time = 0

        #Records
        self.aux = []
        self.solutions = []
        self.solution_time = None

    def fast_solve(self, t_max, exit_type = ORTANT, solver_type = 'BDF', atol=0.000001, rtol=0.001) -> None :
        """
        Solver function, using predefined integrator (default is scipy)
        @param t_max: maximum analog time
        @param exit_type: defines the exit condition (ORTANT = 0) (CONVERGENCE_RADIUS = -1)
        @param solver_type: predefined solver parameter (in scipy or otherwise)
        @param atol, rtol: absolute and relative tolerances
        """
        def exit_ortant(t, y) -> float:
            boolean_sol = [True if elem > 0 else False for elem in y[0:self.problem.number_of_variables]]
            if self.problem.check_solution(boolean_sol):
                #self.solutions.append(boolean_sol)
                #if self.solution_time is None:
                #    self.solution_time = t
                return 0.0
            return -1.0
        exit_ortant.terminal = True
        
        def exit_long(t, y):
            N = self.problem.number_of_variables
            s = y[0:N]
            boolean_sol = [True if elem > 0 else False for elem in s]
            if self.problem.check_solution(boolean_sol):
                sigma = 0.5
                R = sqrt(N-1+sigma**2)
                sabs = np.linalg.norm(s)
                if sabs >= R:
                    return 0
            return -1.0
        exit_long.terminal = True

        def exit_negative_aux(t, y) ->float:
            if any([elem < 1 for elem in y[self.problem.number_of_variables:]]):
                return 0.0
            else:
                return -1.0
        exit_negative_aux.terminal = True

        if exit_type == ORTANT:
            self.sol = solve_ivp(fun=self.problem.rhs,
                            t_span=(0, t_max),
                            y0=self.state,
                            method=solver_type,
                            t_eval=None,
                            dense_output=False,
                            events=exit_ortant,
                            atol=atol,
                            rtol=rtol)
        elif exit_type == CONVERGENCE_RADIUS:
            self.sol = solve_ivp(fun=self.problem.rhs,
                            t_span=(0, t_max),
                            y0=self.state,
                            method=solver_type,
                            t_eval=None,
                            dense_output=False,
                            events=exit_long,
                            atol=atol,
                            rtol=rtol)
        elif exit_type == NEGATIVE_AUX:
            self.sol = solve_ivp(fun=self.problem.rhs,
                            t_span=(0, t_max),
                            y0=self.state,
                            method=solver_type,
                            t_eval=None,
                            dense_output=False,
                            events=exit_negative_aux,
                            atol=atol,
                            rtol=rtol)                    
        else:
            self.sol = solve_ivp(fun=self.problem.rhs,
                            t_span=(0, t_max),
                            y0=self.state,
                            method=solver_type,
                            t_eval=None,
                            dense_output=False,
                            atol=atol,
                            rtol=rtol)
        
    def get_solution(self):
        if self.sol.y.any():
            str_sol = ""
            for elem in [spin_var_series[-1] for spin_var_series in self.sol.y[0:self.problem.number_of_variables]]:
                if elem > 0:
                    str_sol += '1'
                else:
                    str_sol += '0'
            
            return str_sol
        else:
            return None

    def lyapunov_solve(self, t_max, exit_type = 0, solver_type = 'BDF', atol=0.000001, rtol=0.000001) -> None :
        N = self.problem.number_of_variables
        M = self.problem.number_of_clauses

        def exit_ortant(t, y) -> float:
            """CHANGE IT"""
            boolean_sol = [True if elem > 0 else False for elem in y[0:self.problem.number_of_variables]]
            if self.problem.check_solution(boolean_sol):
                sol_index = self.problem.get_solution_index( "".join(['1' if elem else '0' for elem in boolean_sol]) )
                if len(self.solutions) == 0 or self.solutions[-1] != sol_index:
                    self.solutions.append(sol_index)
                return 0
            else:
                return -1.0
        exit_ortant.terminal = True
        def exit_long(t, y):
            N = self.problem.number_of_variables
            s = y[0:N]
            boolean_sol = [True if elem > 0 else False for elem in s]
            if self.problem.check_solution(boolean_sol):
                sigma = 0.5
                R = sqrt(N-1+sigma**2)
                sabs = np.linalg.norm(s)
                if sabs >= R:
                    return 0
            return -1.0

        def extended_system(t, y):
            s = y[:N]
            a = y[N:M]
            #Size N+M square matrix for the tangential space
            U = y[N+M:N+M+(N+M)**2].reshape([N+M, N+M])
            #Size N+M vector for lyapunov exponents
            L = y[N+M+(N+M)**2:2*(N+M)+(N+M)**2]
            f = self.problem.rhs(t, y)
            Df = self.problem.Jakobian(y)
            A = U.T.dot(Df.dot(U))
            dL = np.diag(A).copy()
            for i in range(N+M):
                A[i,i] = 0
                for j in range(i+1,N+M):
                    A[i,j] = -A[j,i]
            dU = U.dot(A)
            return np.concatenate([f,dU.flatten(),dL])

        y0 = self.state
        U0 = np.identity(N+M)
        L0 = np.zeros(N+M)
        initial_state = np.concatenate([y0, U0.flatten(), L0])

        if exit_type == ORTANT:
            self.sol = solve_ivp(fun=extended_system,
                            t_span=(0, t_max),
                            y0=initial_state,
                            method=solver_type,
                            t_eval=None,
                            dense_output=False,
                            events=exit_ortant,
                            atol=atol,
                            rtol=rtol)
        if exit_type == CONVERGENCE_RADIUS:
            self.sol = solve_ivp(fun=self.problem.rhs,
                            t_span=(0, t_max),
                            y0=initial_state,
                            method=solver_type,
                            t_eval=None,
                            dense_output=False,
                            events=exit_long,
                            atol=atol,
                            rtol=rtol)

