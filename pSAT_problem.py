#Boolean satisfiability problem solver with continuous dynamical system
import numpy as np
import matplotlib.pyplot as plt

class SATProblem:
    def __init__(self, cnf_file_name):
        with open(cnf_file_name) as cnf_file:
            lines = cnf_file.readlines()
            self.number_of_variables = int(lines[0].split(' ')[2])
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
            if variable == j:
                return 1
            elif variable == -j:
                return -1
        return 0

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

class CTDSolver:
    def __init__(self, SATproblem, integrator) -> None:
        self.SATproblem = SATproblem
        self.integrator = integrator
        self.s = np.random.rand( self.SATproblem.number_of_variables ) - np.array( [0.5 for i in range(self.SATproblem.number_of_variables)] )
        #self.a = np.random.rand( self.SATproblem.number_of_clauses )
        self.a = np.array( [1 for i in range(self.SATproblem.number_of_clauses)] )
        self.aux = []
        self.traj = []
        self.diff_vecs = []

    def K_m1(self, m, s):
        """Depricated"""
        return np.prod([( 1-self.SATproblem.c_mj(m,j) * s[j] ) for j in range(self.SATproblem.number_of_variables)])
    
    def K_mi_m(self, m, i, s):
        K_mi = self.K_mi(m, i, s)
        return K_mi*K_mi*(1-self.SATproblem.c_mj(m,i)*s[i])

    def K_mi(self, m, i, s):
        return pow(2, -self.SATproblem.number_of_literals[m]) * np.prod([( 1-self.SATproblem.c_mj(m,j) * s[j] ) for j in range(self.SATproblem.number_of_variables) if i != j])
    
    def gradV_i(self, i, s):
        return sum([2*self.a[m]*self.SATproblem.c_mj(m, i)*self.K_mi_m(m, i, s) for m in range(self.SATproblem.number_of_clauses) ])

    def mgradV(self, s):
        return np.array([self.gradV_i(i, s) for i in range(self.SATproblem.number_of_variables)])

    def x(self):
        return [True if elem > 0 else False for elem in self.s]

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
            test_solution = self.x()
        else:
            test_solution = sol
        for i, row in enumerate(self.SATproblem.clauses):
            if not check_row(row, test_solution):
                incorrect_flag = True
                break

        if incorrect_flag:
            return False
        else:
            return True

    def save_states(self):
        self.aux.append(self.a)
        self.traj.append(self.s)

    def step_function(self):
        new_s = self.integrator.step(self.s, lambda s_ : self.mgradV(s_))
        new_a = self.integrator.step(self.a, lambda a : np.array([a[m] * self.K_m1(m, self.s) for m in range(self.SATproblem.number_of_clauses)]) )
        diff_vec = self.s-new_s
        #Adaptive step size
        if (np.linalg.norm(diff_vec) < 0.001 and not self.integrator.h > 0.1):
            self.integrator.h *= 2
        elif (np.linalg.norm(diff_vec) > 0.01 and not self.integrator.h < 0.0001):
            self.integrator.h /= 2

        self.diff_vecs.append(diff_vec)
        self.s = new_s
        self.a = new_a

    def solve(self, max_steps = 6000):
        for i in range(max_steps):
            self.step_function()
            if i%10 == 0:
                self.save_states()
                if self.check_solution():
                    break
            if i%2000 == 0 and i != 0:
                print("Restart")
                self.aux = []
                self.traj = []
                self.s = np.random.rand( self.SATproblem.number_of_variables ) - np.array( [0.5 for i in range(self.SATproblem.number_of_variables)] )
                self.a = np.array( [1 for i in range(self.SATproblem.number_of_clauses)] )

    def plot_traj(self):
        plt.grid(True)
        plt.title('Spin variables')
        for i in range(self.SATproblem.number_of_variables):
            x = [j for j in range(len(self.traj))]
            y = [self.traj[j][i] for j in range(len(self.traj))]
            plt.plot(x, y, label=str(i+1))
        plt.legend()
        plt.show()
    
    def plot_aux(self):
        plt.grid(True)
        plt.title("Aux variables")
        for i in range(self.aux[0].size):
            x = [j for j in range(len(self.aux))]
            y = [self.aux[j][i] for j in range(len(self.aux))]
            plt.plot(x, y, label=str(i+1))
        #plt.legend()
        plt.show()