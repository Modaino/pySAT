#########################################
#                                       #
# Boolean satisfiability problem solver #
#  with continuous dynamical system     #
#                                       #
# Written by Áron Vízkeleti             #
#       on 2021-10-10                   #
#       last modified 2023-06-01        #
#                                       #
#########################################

from cmath import nan
from math import sqrt, pi, sin
from xmlrpc.client import Boolean
import numpy as np
from abc import abstractclassmethod
from random import sample, randint, random, getrandbits
from scipy.integrate import solve_ivp
from ctypes import CDLL, POINTER, c_double, c_int

#Constants

ORTANT = 0
CONVERGENCE_RADIUS = -1
NEGATIVE_AUX = -2
HYPER_SPHERE = -3

RHS_TYPE_ONE = 1 # Original CTDS
RHS_TYPE_TWO = 2 # CTDS with K_m squred
RHS_TYPE_THREE = 3 # CTDS with K_m squared and central potential
RHS_TYPE_FOUR = 4
RHS_TYPE_FIVE = 5
RHS_TYPE_SIX = 6 #Recurrance prevention
RHS_TYPE_SEVEN = 7 #Exponential memory supression with z
RHS_TYPE_EIGHT = 8 #Exponential memory supression
RHS_TYPE_NINE = 9 #No aux of any kind
RHS_TYPE_TEN = 10 #Second order memory, inefficient
RHS_TYPE_ELEVEN = 11 #Second order memory

BIPARTITE_PLOT = 101
SPRING_PLOT = 102

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
    def __init__(self, cnf_file_name, so_file_name, n = 15, alpha = 4.264, literal_number = 3, rhs_type = RHS_TYPE_ONE, planted = 3, lmbd = 0.25):
        """
        Constructor
        @param cnf_file_name: cnf-file defining the problem, if set to None, generates a random problem
        @param so_file_name: c/c++ library containing (hopefully fast) implementation of rhs and jakobian matrices
        @param n: optional, number of variables in randomly generated problem
        @param alpha: optional, ration of clauses (w.r.t n) in randomly generated problem
        @param literal_number: optional, defines the length of clauses (default is 3)
        @param rhs_type: optional, selects type of rhs (RHS_TYPE_ONE = 1) (RHS_TYPE_TWO = 2) etc.
        """
        #Misc. init
        self.valid_solutions = None
        self.rhs_type = rhs_type
        self.alpha = None
        self.lmdb = lmbd
        if self.rhs_type == RHS_TYPE_SIX or self.rhs_type == RHS_TYPE_SEVEN:
            self.tried_ortants = []
            self.tried_ortant_idx = 0
            self.rec_prev_factor = 1.0

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
            if planted:
                self.clauses = self.generate_planted_problem(n, self.number_of_clauses, planted, literal_number)
            else:
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
            #self.cSAT_functions.rhs6.restype = None
            self.cSAT_functions.rhs7.restype = None
            self.cSAT_functions.rhs7.argtypes = [c_int, c_int, c_double, POINTER(c_int), POINTER(c_double), POINTER(c_double)]
            self.cSAT_functions.rhs8.restype = None
            self.cSAT_functions.rhs8.argtypes = [c_int, c_int, c_double, POINTER(c_int), POINTER(c_double), POINTER(c_double)]
            self.cSAT_functions.rhs9.restype = None
            self.cSAT_functions.rhs10.restype = None
            self.cSAT_functions.rhs11.restype = None
            # For debugging only
            self.cSAT_functions.jacobian1.restype = None
            self.cSAT_functions.jacobian2.restype = None
            self.cSAT_functions.K_m.restype = c_double
            self.cSAT_functions.K_m.argtypes = [c_int, POINTER(c_double), POINTER(c_int), c_int]


            clause_matrix = self.c.flatten().astype(np.int32) # c
            self.clause_matrix_pointer = clause_matrix.ctypes.data_as(POINTER(c_int))

        #Setting up the rhs funciton
        if not self.cSAT_functions:
            if self.rhs_type == RHS_TYPE_ONE:
                self.rhs = self.rhs_type_one_py
            elif self.rhs_type == RHS_TYPE_TWO:
                self.rhs = self.rhs_type_two_py
            elif self.rhs_type == RHS_TYPE_THREE:
                self.rhs = self.rhs_type_three_py
            elif self.rhs_type == RHS_TYPE_FOUR:
                self.rhs = self.rhs_type_four_py
            elif self.rhs_type == RHS_TYPE_FIVE:
                self.rhs = self.rhs_type_five_py
            elif self.rhs_type == RHS_TYPE_SIX:
                self.rhs = self.rhs_type_six_py
            elif self.rhs_type == RHS_TYPE_SEVEN:
                self.rhs = self.rhs_type_seven_py
            elif self.rhs_type == RHS_TYPE_EIGHT:
                self.rhs = self.rhs_type_eight_py
            elif self.rhs_type == RHS_TYPE_NINE:
                self.rhs = self.rhs_type_nine_py
            elif self.rhs_type == RHS_TYPE_TEN:
                self.rhs = self.rhs_type_ten_py
        else:
            if self.rhs_type == RHS_TYPE_ONE:
                self.rhs = self.rhs_type_one_c
            elif self.rhs_type == RHS_TYPE_TWO:
                self.rhs = self.rhs_type_two_c
            elif self.rhs_type == RHS_TYPE_THREE:
                self.rhs = self.rhs_type_three_c
            elif self.rhs_type == RHS_TYPE_FOUR:
                self.rhs = self.rhs_type_four_c
            elif self.rhs_type == RHS_TYPE_FIVE:
                self.rhs = self.rhs_type_five_c
            elif self.rhs_type == RHS_TYPE_SIX:
                self.rhs = self.rhs_type_six_c
            elif self.rhs_type == RHS_TYPE_SEVEN:
                self.rhs = self.rhs_type_seven_c
            elif self.rhs_type == RHS_TYPE_EIGHT:
                self.rhs = self.rhs_type_eight_c
            elif self.rhs_type == RHS_TYPE_NINE:
                self.rhs = self.rhs_type_eight_c
            elif self.rhs_type == RHS_TYPE_TEN:
                self.rhs = self.rhs_type_ten_c
            elif self.rhs_type == RHS_TYPE_ELEVEN:
                self.rhs = self.rhs_type_eleven_c

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
            state = y.astype(np.double) # s & a
            state_pointer = state.ctypes.data_as(POINTER(c_double))

            result = np.empty((self.number_of_variables + self.number_of_clauses)**2)
            result = result.astype(np.double) # (s + a)**2
            result_pointer = result.ctypes.data_as(POINTER(c_double))
        
            if self.rhs_type == RHS_TYPE_ONE:
                self.cSAT_functions.jacobian1(self.number_of_variables, self.number_of_clauses, self.clause_matrix_pointer, state_pointer, result_pointer)
            elif self.rhs_type == RHS_TYPE_TWO:
                raise NotImplementedError
                #self.cSAT_functions.jacobian2(self.number_of_variables, self.number_of_clauses, self.clause_matrix_pointer, state_pointer, result_pointer)
            return result.reshape([N_+M_, N_+M_])

    def rhs(self, t, y):
        """Right-hand side of the differential equation defining the system, to be overwritten in runtime"""
        pass

    def K(self, m, s):
        """Clause term, as defined in nature physics letter doi:10.1038/NPHY2105"""
        return pow(2, -self.number_of_literals[m])*np.prod([( 1-self.c[m,j] * s[j] ) for j in range(self.number_of_variables)])

    def k(self, m, i, s):
        """Modified clause term, as defined in nature physics letter doi:10.1038/NPHY2105"""
        return pow(2, -self.number_of_literals[m])*np.prod([( 1-self.c[m,j] * s[j] ) for j in range(self.number_of_variables) if i != j])

    def remove_variable(self, variable) -> None:
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

        #Resetting variables
        self.clauses = new_clauses
        self.number_of_variables -= 1
        self.number_of_clauses = len(new_clauses)
        self.number_of_literals = new_literals
        self.c = np.array([[1 if (j+1) in clause else -1 if -(j+1) in clause else 0 for j in range(self.number_of_variables) ] for clause in self.clauses])
        self.alpha = self.number_of_clauses/self.number_of_variables

    def smallest_variable(self):
        """Returns the index of the varibale that appears in the smallest number of clauses"""
        used_in = [0*1 for i in range(self.number_of_variables)]
        for clause in self.clauses:
            for elem in clause:
                used_in[abs(elem)-1] += 1
        #print(used_in)
        return used_in.index(min(used_in))+1

    def generate_planted_problem(self, N, M, S, literal_number):
        """Generates a sat problem with constant clause length, with planted solutions. Note that there can be 'unplanted' solutions.
        @param literal_number: The length of each clause
        @param N: Number of variables
        @param M: Number of clauses
        @param S: Number of planted solutions
        """
        def compatible(clause, solutions):
            if clause is None:
                return False
            for sol in solutions:
                is_compatible = False
                for elem in clause:
                    if elem > 0:
                        if sol[elem-1]:
                            is_compatible = True
                    if elem < 0:
                        if not sol[(-elem)-1]:
                            is_compatible = True
                # If the clause is not compatible with this solution
                if not is_compatible:
                    return False
            # If it was compatible with all the solutions
            return True
        
        # Get the sols matrix as an input (not as random)
        sols = np.zeros((S, N))
        for i in range(S):
            for j in range(N):
                if getrandbits(1):
                    sols[i,j] = 1

        clauses = []
        for m in range(M):
            test_clause = None
            while not compatible(test_clause, sols):
                test_clause = [elem if randint(0,1) else -elem for elem in sample(range(1, N+1), literal_number)]
            clauses.append(test_clause)
        
        self.planted_solutions = ["".join([str(int(elem)) for elem in sol]) for sol in sols]
        return clauses

    def downconvert_4_3(self):
        """Converts a 4SAT problem into a 3SAT problem"""
        new_clauses = []
        N = self.number_of_variables
        
        for (idx, clause) in enumerate(self.clauses):
            new_clause_1 = [clause[0], clause[1], N+1+idx]
            new_clause_2 = [clause[2], clause[3], -(N+1+idx)]
            new_clauses.append(new_clause_1)
            new_clauses.append(new_clause_2)
            
        self.number_of_variables += self.number_of_clauses
        self.number_of_clauses += self.number_of_clauses
        self.clauses = new_clauses
        self.valid_solutions = None

    def down_convert_clause(self, clause_idx):
        """Converts a clause of length greater than 3 to a smaller clause. NOT FINISHED"""
        old_clause = self.clauses[clause_idx]
        k = len(old_clause)
        N = self.number_of_variables

        if k <= 3:
            raise ValueError("Cannot decrese clause length below 3")

        # k-3 new variables
        new_clauses = []
        new_clauses.append([old_clause[0], old_clause[1], N+1])
        for j in range(1, k-3):
            new_clauses.append([-(N+j)])

    def write_problem_to_file(self, name) -> None:
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
        """Returns the ratio of clauses to variables, and saves it into a member variable if it is not saved already."""
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
            if self.rhs_type == RHS_TYPE_SIX or self.rhs_type == RHS_TYPE_SEVEN:
                self.set_add(solution)
            return False #Solution does not solve the sat problem
        else:
            #self.solution = "".join(['1' if elem else '0' for elem in test_solution])
            return True #Solutions solves the problem

    def get_number_of_satisfied_clauses(self, discrete_state):
        """Returns the number of satisfied clauses given a state in discretized form (list of plus minus ones)"""
        result = 0
        for clause in self.clauses:
            for literal in clause:
                variable_idx = abs(literal)
                expected_truth_value = 1 if (literal > 0) else -1
                if expected_truth_value ==  discrete_state[variable_idx-1]:
                    result += 1
                    break
        return result

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

    def set_add(self, solution):
        solution_vec = np.array([1 if spin is True else -1 for spin in solution])
        if len(self.tried_ortants) == 0:
            self.tried_ortants.append( solution_vec )
        else:
            for elem in self.tried_ortants:
                if (elem == solution_vec).all():
                    return None
            if len(self.tried_ortants) < 100:
                self.tried_ortants.append( solution_vec )
            else:
                self.tried_ortants[self.tried_ortant_idx] = solution_vec
                if self.tried_ortant_idx < 100:
                    self.tried_ortant_idx += 1
                else:
                    self.tried_ortant_idx = 0
            return None
            
    def harden_clause(self, clause_idx, solutions):
        """Only one literal needs to be true per clause, this function flips the first of the excess true literals in a clause if there is one"""
        literal_satisfaction = [False for k in range(self.number_of_literals[clause_idx])]

        for k in range(self.number_of_literals[clause_idx]):
            # Is the literal true for all the given solutions?
            variable_idx = abs(self.clauses[clause_idx][k])
            literal_value = '1' if self.clauses[clause_idx][k] > 0 else '0'

            literal_satisfaction[k] = all(sol[variable_idx-1] == literal_value for sol in solutions)
        
        # Flipping the first excess literal
        if literal_satisfaction.count(True) >= 2:
            for k in range(self.number_of_literals[clause_idx]):
                if literal_satisfaction[k]:
                    self.clauses[clause_idx][k] = -self.clauses[clause_idx][k]
                    # for sol in solutions:
                    #     if not self.check_solution(sol):
                    #         self.clauses[clause_idx][k] = -self.clauses[clause_idx][k]
                    #         return False
                    return True
        else:
            return False
            
    # RHS funcitons
    
    def rhs_type_one_c(self, t, y):
        state = y.astype(np.double) # s & a
        state_pointer = state.ctypes.data_as(POINTER(c_double))
        result = np.empty(self.number_of_variables + self.number_of_clauses)
        result = result.astype(np.double) # empty vector of size (s & a)
        result_pointer = result.ctypes.data_as(POINTER(c_double))
        self.cSAT_functions.rhs1(self.number_of_variables, self.number_of_clauses, self.clause_matrix_pointer, state_pointer, result_pointer)
        return result

    def rhs_type_one_py(self, t, y):
        N_ = self.number_of_variables
        s = y[:N_]
        a = y[N_:]
        ds = np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) for i in range(self.number_of_variables) ])
        da = np.array([a[m]*self.K(m, s) for m in range(self.number_of_clauses)])
        return np.concatenate((ds, da), axis=None)

    def rhs_type_two_c(self, t, y):
        state = y.astype(np.double) # s & a
        state_pointer = state.ctypes.data_as(POINTER(c_double))
        result = np.empty(self.number_of_variables + self.number_of_clauses)
        result = result.astype(np.double) # empty vector of size (s & a)
        result_pointer = result.ctypes.data_as(POINTER(c_double))
        self.cSAT_functions.rhs2(self.number_of_variables, self.number_of_clauses, self.clause_matrix_pointer, state_pointer, result_pointer)
        return result

    def rhs_type_two_py(self, t, y):
        N_ = self.number_of_variables
        s = y[:N_]
        a = y[N_:]
        ds = np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) for i in range(self.number_of_variables) ])
        da = np.array([a[m]*(self.K(m, s)**2) for m in range(self.number_of_clauses)])
        return np.concatenate((ds, da), axis=None)

    def rhs_type_three_c(self, t, y):
        state = y.astype(np.double) # s & a
        state_pointer = state.ctypes.data_as(POINTER(c_double))
        result = np.empty(self.number_of_variables + self.number_of_clauses)
        result = result.astype(np.double) # empty vector of size (s & a)
        result_pointer = result.ctypes.data_as(POINTER(c_double))
        self.cSAT_functions.rhs3(self.number_of_variables, self.number_of_clauses, self.clause_matrix_pointer, state_pointer, result_pointer)
        return result

    def rhs_type_three_py(self, t, y):
        N_ = self.number_of_variables
        s = y[:N_]
        a = y[N_:]
        b = 0.0725
        a_ = sum(a)/self.number_of_clauses
        constant = 0.5*pi*b*self.alpha * a_
        ds = np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) + constant*sin(pi*s[i])  for i in range(self.number_of_variables) ])
        da = np.array([a[m]*(self.K(m, s)**2) for m in range(self.number_of_clauses)])
        return np.concatenate((ds, da), axis=None)

    def rhs_type_four_c(self, t, y):
        state = y.astype(np.double) # s & a
        state_pointer = state.ctypes.data_as(POINTER(c_double))
        result = np.empty(self.number_of_variables + self.number_of_clauses)
        result = result.astype(np.double) # empty vector of size (s & a)
        result_pointer = result.ctypes.data_as(POINTER(c_double))
        self.cSAT_functions.rhs4(self.number_of_variables, self.number_of_clauses, self.clause_matrix_pointer, state_pointer, result_pointer)
        return result

    def rhs_type_four_py(self, t, y):
        N_ = self.number_of_variables
        s = y[:N_]
        a = y[N_:]
        b = 0.0725
        a_ = sum(a)/len(a)
        constant = 0.5*pi*b*self.alpha * a_
        ds = np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) + constant*sin(pi*s[i])  for i in range(self.number_of_variables) ])
        da = np.array([a[m]*(self.K(m, s)) for m in range(self.number_of_clauses)])
        return np.concatenate((ds, da), axis=None)

    def rhs_type_five_c(self, t, y):
        state = y.astype(np.double) # s & a
        state_pointer = state.ctypes.data_as(POINTER(c_double))
        result = np.empty(self.number_of_variables + self.number_of_clauses)
        result = result.astype(np.double) # empty vector of size (s & a)
        result_pointer = result.ctypes.data_as(POINTER(c_double))
        self.cSAT_functions.rhs5(self.number_of_variables, self.number_of_clauses, self.clause_matrix_pointer, state_pointer, result_pointer)
        return result

    def rhs_type_five_py(self, t, y):
        N_ = self.number_of_variables
        s = y[:N_]
        a = y[N_:]
        b = 0.0725
        a_ = sum(a)/len(a)
        constant = 0.5*pi*b*self.alpha * a_
        ds = (-1)*np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) + constant*sin(pi*s[i])  for i in range(self.number_of_variables) ])
        da = (-1)*np.array([a[m]*(self.K(m, s)) for m in range(self.number_of_clauses)])
        return np.concatenate((ds, da), axis=None)

    def rhs_type_six_c(self, t, y):
        state = y.astype(np.double) # s & a
        state_pointer = state.ctypes.data_as(POINTER(c_double))
        result = np.empty(self.number_of_variables + self.number_of_clauses)
        result = result.astype(np.double) # empty vector of size (s & a)
        result_pointer = result.ctypes.data_as(POINTER(c_double))
        self.cSAT_functions.rhs6(self.number_of_variables, self.number_of_clauses, self.clause_matrix_pointer, state_pointer, result_pointer)
        return result

    def rhs_type_six_py(self, t, y):
        N_ = self.number_of_variables
        s = y[:N_]
        a = y[N_:]
        if len(self.tried_ortants) > 2:
            L = np.array(sum([s - sl for sl in self.tried_ortants]))
            L = (self.rec_prev_factor / (np.linalg.norm(L)**3)) * L
        else:
            L = np.zeros(self.number_of_variables)
        b = 0.0725
        a_ = sum(a)/self.number_of_clauses
        constant = 0.5*pi*b*self.alpha * a_
        ds = L + np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) + constant*sin(pi*s[i])  for i in range(self.number_of_variables) ])
        da = np.array([a[m]*(self.K(m, s)**2) for m in range(self.number_of_clauses)])
        return np.concatenate((ds, da), axis=None)

    def rhs_type_seven_c(self, t, y):
        state = y.astype(np.double) # s & a
        state_pointer = state.ctypes.data_as(POINTER(c_double))
        result = np.empty(self.number_of_variables + self.number_of_clauses)
        result = result.astype(np.double) # empty vector of size (s & a)
        result_pointer = result.ctypes.data_as(POINTER(c_double))
        self.cSAT_functions.rhs7(self.number_of_variables, self.number_of_clauses, self.lmdb, self.clause_matrix_pointer, state_pointer, result_pointer)
        return result

    def rhs_type_seven_py(self, t, y):
        N_ = self.number_of_variables
        s = y[:N_]
        z = y[N_:]
        ds = np.array([sum( 2*[np.exp(z[m])*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) for i in range(self.number_of_variables) ])
        dz = np.array([(self.K(m, s) - self.lmdb * z[m]) for m in range(self.number_of_clauses)])
        return np.concatenate((ds, dz), axis=None)

    def rhs_type_eight_c(self, t, y):
        state = y.astype(np.double) # s & a
        state_pointer = state.ctypes.data_as(POINTER(c_double))
        result = np.empty(self.number_of_variables + self.number_of_clauses)
        result = result.astype(np.double) # empty vector of size (s & a)
        result_pointer = result.ctypes.data_as(POINTER(c_double))
        self.cSAT_functions.rhs8(self.number_of_variables, self.number_of_clauses, self.lmdb, self.clause_matrix_pointer, state_pointer, result_pointer)
        return result

    def rhs_type_eight_py(self, t, y):
        N_ = self.number_of_variables
        s = y[:N_]
        a = y[N_:]
        ds = np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) for i in range(self.number_of_variables) ])
        da = np.array([a[m]*(self.K(m, s)**2 - self.lmdb * np.log(a[m])) for m in range(self.number_of_clauses)])
        return np.concatenate((ds, da), axis=None)

    def rhs_type_nine_c(self, t, y):
        state = y.astype(np.double) # s & a
        state_pointer = state.ctypes.data_as(POINTER(c_double))
        result = np.empty(self.number_of_variables + self.number_of_clauses)
        result = result.astype(np.double) # empty vector of size (s & a)
        result_pointer = result.ctypes.data_as(POINTER(c_double))
        self.cSAT_functions.rhs9(self.number_of_variables, self.number_of_clauses, self.clause_matrix_pointer, state_pointer, result_pointer)
        return result

    def rhs_type_nine_py(self, t, y):
        N_ = self.number_of_variables
        s = y[:N_]
        a = y[N_:]
        ds = np.array([sum(2*[a[m]*self.c[m, i]* (1-self.c[m, i]*s[i]) *(self.k(m, i, s)**2) for m in range(self.number_of_clauses)]) for i in range(self.number_of_variables) ])
        da = np.zeros(self.number_of_clauses)
        return np.concatenate((ds, da), axis=None)

    def rhs_type_ten_py(self, t, y):
        N_ = self.number_of_variables
        M_ = self.number_of_clauses
        s = y[:N_]
        b = y[N_:]

        ds = np.empty(N_)
        for i in range(N_):
            summ = s[i]
            for m in range(M_):
                for n in range(M_):
                    summ += b[m*M_ + n] * ( self.c[m,i] * (1-s[i]*self.c[m, i])*pow(self.k(m, i, s), 2) + self.c[n, i] * (1-s[i]*self.c[n, i])*pow(self.k(n, i, s), 2))
            ds[i] = summ
        
        db = np.array([b[i] * self.K(int(i/M_), s) * self.K(i % M_, s) for i in range(M_*M_)])

        return np.concatenate((ds, db), axis=None)

    def rhs_type_ten_c(self, t, y):
        state = y.astype(np.double) # s & b
        state_pointer = state.ctypes.data_as(POINTER(c_double))
        result = np.zeros(self.number_of_variables + self.number_of_clauses*self.number_of_clauses)
        result = result.astype(np.double) # empty vector of size (N+M^2)
        result_pointer = result.ctypes.data_as(POINTER(c_double))
        self.cSAT_functions.rhs10(self.number_of_variables, self.number_of_clauses, self.clause_matrix_pointer, state_pointer, result_pointer)
        return result

    def rhs_type_eleven_c(self, t, y):
        state = y.astype(np.double) # s & b_flattened
        state_pointer = state.ctypes.data_as(POINTER(c_double))
        result = np.empty(self.number_of_variables + int(self.number_of_clauses*(self.number_of_clauses+1)/2))
        result = result.astype(np.double) # empty vector of size (N+M+M^2)
        result_pointer = result.ctypes.data_as(POINTER(c_double))
        self.cSAT_functions.rhs11(self.number_of_variables, self.number_of_clauses, self.clause_matrix_pointer, state_pointer, result_pointer)
        return result


#Numerical solver definition(s)

class CTD:
    def __init__(self, problem, integrator = None, initial_s = None, random_aux = False, initial_aux = None, initial_baux = None) -> None:
        """Constructor for the Continuous Time SAT solver"""
        
        self.problem = problem
        N = problem.number_of_variables
        M = problem.number_of_clauses

        #Dynamical variables
        if self.problem.rhs_type < RHS_TYPE_TEN:
            self.state = np.ones(N+M)
            if random_aux == True:
                self.state[N:] = np.array( [random()*15 for i in range(M)] )
            elif initial_aux is not None:
                self.state[N:] = initial_aux
        else:
            if self.problem.rhs_type == RHS_TYPE_TEN:
                self.state = np.ones(N + M*M)
            elif self.problem.rhs_type == RHS_TYPE_ELEVEN:
                self.state = np.ones(N + int(M*(M+1)/2))

            if initial_baux is not None:
                self.state[N:] = initial_baux
            

        if initial_s is None:
            self.state[0:N] = 2*np.random.rand(N) - np.ones(N)
        else:
            self.state[0:N] = initial_s

        #Records
        self.time = 0
        self.aux = []
        self.solutions = []
        self.solution_time = None
        self.inside_hypersphere = None
        self.sol_length = None

    def fast_solve(self, t_max, exit_type = ORTANT, solver_type = 'BDF', atol=0.0001, rtol=0.001, hypersphere_radius = 1.0) -> None :
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

        def exit_hypersphere(t ,y) -> float:
            s = y[:self.problem.number_of_variables]
            if np.linalg.norm(s, 2) >= hypersphere_radius:
                return 1.0
            else:
                return -1.0
        exit_hypersphere.terminal = False

        def enter_hypersphere(t ,y) -> float:
            s = y[:self.problem.number_of_variables]
            if np.linalg.norm(s, 2) < hypersphere_radius:
                return 1.0
            else:
                return -1.0
        enter_hypersphere.terminal = False

        def exit_hypercube(t, y) -> float:
            for coord in y[:self.problem.number_of_variables]:
                if coord < 1:
                    return 1.0
                return -1.0
        exit_hypercube.terminal = True
            

        if exit_type == ORTANT:
            self.sol = solve_ivp(fun=self.problem.rhs,
                            t_span=(0, t_max),
                            y0=self.state,
                            method=solver_type,
                            t_eval=None,
                            dense_output=False,
                            events=[exit_ortant,exit_hypercube],
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
        elif exit_type == HYPER_SPHERE:
            if np.linalg.norm(self.state[:self.problem.number_of_variables],2) < hypersphere_radius:
                self.inside_hypersphere = True
            else:
                self.inside_hypersphere = False
            self.sol = solve_ivp(fun=self.problem.rhs,
                            t_span=(0, t_max),
                            y0=self.state,
                            method=solver_type,
                            t_eval=None,
                            dense_output=False,
                            events=exit_hypersphere,
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

    def get_satisfied_C_of_t(self):
        if not self.sol:
            print("Solution object not yet generated")
            return None
        else:
            C_of_t = []
            for elem in np.transpose(self.sol.y):
                solution_vec = np.array([1 if spin > 0 else -1 for spin in elem[:self.problem.number_of_variables]])
                C_of_t.append(self.problem.get_number_of_satisfied_clauses(solution_vec))
            return (self.sol.t, C_of_t)

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

    def get_sol_time(self):
        if self.solution_time is not None:
            return self.solution_time
        elif self.sol is not None:
            return self.sol.t[-1]
        else:
            return None

    def get_sol_length(self):
        """Returns a numpy list of cummulative trajectory lengths"""
        if self.sol is not None:
            if self.sol_length is None:
                N = self.problem.number_of_variables
                nbr_points = len(self.sol.t)
                result = np.zeros(nbr_points)
                for idx in range(nbr_points):
                    if idx < nbr_points-1:
                        p2 = self.sol.y[0:N, idx+1]
                        p1 = self.sol.y[0:N, idx]
                        sum_of_squares = np.sum(np.square(p2-p1))
                        result[idx+1] = result[idx] + np.sqrt(sum_of_squares) 
                self.sol_length = result
            return self.sol_length
        else:
            return None

    def get_ortant_path(self):
        pass
        # if self.sol is None:
        #     print("There is no solution attempt")
        #     return None
        # else:


class Warning_propagator:
    def __init__(self, SAT_problem) -> None:
        import networkx as nx

        self.problem = SAT_problem
        self.M = SAT_problem.number_of_clauses
        self.N = SAT_problem.number_of_variables
        self.g = nx.Graph()
        self.g.add_nodes_from([(i, {'type' : 'variable'}) for i in range(self.N)], bipartite=0) # variable nodes
        self.g.add_nodes_from([i+self.N for i in range(self.M)], bipartite=1)                   # clause nodes
        for m, clause in enumerate(SAT_problem.clauses):
            for literal in clause:
                wrng = Warning()
                wrng.random_init()
                if literal > 0:
                    self.g.add_edge(literal-1, m+self.N, weight=1, warning=wrng)
                else:
                    self.g.add_edge(-literal-1, m+self.N, weight=-1, warning=wrng)

    def propagate(self, N_max):
        for n in range(N_max):
            pass

    def plot_factorgraph(self, plot_algorithm):
        import matplotlib.pyplot as plt
        import networkx as nx
        from networkx.algorithms import bipartite

        pos = dict()

        if plot_algorithm == BIPARTITE_PLOT:
            pos.update( (i, (1, i)) for i in range(self.N) ) # put nodes from X at x=1
            pos.update( (i, (2, i-self.N)) for i in range(self.N, self.N+self.M) ) # put nodes from Y at x=2
        elif plot_algorithm == SPRING_PLOT:
            pos = nx.spring_layout(self.g)

        nx.draw_networkx_nodes(self.g, pos=pos, nodelist=range(self.N), node_shape='s', node_color='purple')
        nx.draw_networkx_nodes(self.g, pos=pos, nodelist=range(self.N, self.N+self.M), node_shape='o', node_color='blue')
        nx.draw_networkx_labels(self.g, pos)
        my_edge_color = ['r' if self.g.get_edge_data(u, v)['weight'] > 0 else 'k' for u, v, in self.g.edges()]
        nx.draw_networkx_edges(self.g, pos, edge_color=my_edge_color)
        plt.show()

# Miscelanious classes

class Message:
    def __init__(self, value = None) -> None:
        self.value = value

    def print(self):
        print(self.value)
    
    def step_fnc(self, x):
        return 0 if x <= 0 else 1

class Warning( Message ):
    def __init__(self, value = None) -> None:
        super().__init__(value)

    def random_init(self):
        self.value = randint(0, 1)

    def update(self, graph, clause_node_a, target_variable_node_i) -> None:
        cavity_fields = []
        Jj = []
        neighbours = graph.neighbors(clause_node_a)
        V_plus = [neighbour_j for neighbour_j in neighbours if graph.get_edge_data(clause_node_a, neighbour_j)['weight'] > 0 and neighbour_j != target_variable_node_i]
        V_minus = [neighbour_j for neighbour_j in neighbours if neighbour_j not in V_plus and neighbour_j != target_variable_node_i]

        for neighbour_j in graph.neighbors(clause_node_a):
            if neighbour_j != target_variable_node_i:
                Jj.append(1 if graph.get_edge_data(neighbour_j, clause_node_a)['weight'] > 0 else -1)
                cavity_fields.append(sum([graph.get_edge_data(b, neighbour_j)['warning'].value for b in V_plus])-sum([graph.get_edge_data(b, neighbour_j)['warning'].value for b in V_minus]))
    
        self.value = np.prod( np.array([self.step_fnc(cavity_fields[j] * Jj[j]) for j in len(cavity_fields)]) )
