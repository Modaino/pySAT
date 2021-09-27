#Boolean satisfiability problem solver with continuous dynamical system
import numpy as np

class SATProblem:
    def __init__(self, cnf_file_name):
        with open(cnf_file_name) as cnf_file:
            lines = cnf_file.readlines()
            self.number_of_variables = int(lines[0].split(' ')[2])
            self.number_of_clauses = int(lines[0].split(' ')[3])

            #(v)^(v)#
            clause_and = []
            for i in range(self.number_of_clauses + 1):
                if i > 0:
                    clause_or = []
                    for variable_str in lines[i].split(' '):
                        variable = int(variable_str)
                        if variable != 0:
                            clause_or.append(variable)
                    clause_and.append(clause_or)

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
        self.s = 2*np.random.rand( self.SATproblem.number_of_variables ) - np.array( [1 for i in range(self.SATproblem.number_of_variables)] )
        self.a = np.random.rand( self.SATproblem.number_of_clauses )

    def K_m(self, m, s):
        return np.prod([( 1-self.SATproblem.c_mj(m,j) * s[j] ) for j in range(self.SATproblem.number_of_variables)])
    
    def K_mi(self, m, i, s):
        #prod = 1.0
        #for j in range(self.SATproblem.number_of_variables):
        #    if j != i:
        #        prod *= ( 1-self.SATproblem.c_mj(m,j) * s[j] )
        #return prod
        return self.K_m(m, s)/( 1-self.SATproblem.c_mj(m,i) * s[i] )
    
    def V(self):
        return sum([self.a[m]*self.K_m(m) for m in range(self.SATproblem.number_of_clauses)])

    def gradV_i(self, i, s):
        print(str(i) + " out of " + str(self.SATproblem.number_of_variables))
        return sum([2*self.a[m]*self.SATproblem.c_mj(m, i)*self.K_mi(m, i, s)*self.K_m(m, s) for m in range(self.SATproblem.number_of_clauses) ])

    def gradV(self):
        vecK = np.array([self.K_m(m, self.s) for m in range(self.SATproblem.number_of_clauses)])
        print('yey')
        matK = np.array([[self.K_mi(m, i, self.s) for i in range(self.SATproblem.number_of_variables) ] for m in range(self.SATproblem.number_of_clauses)])
        print('yey')
        return np.dot(self.a, np.multiply(self.SATproblem.c, matK)) + vecK

    def mgradV(self, s):
        return (-1)*np.array([self.gradV_i(i, s) for i in range(self.SATproblem.number_of_variables)])

    def x(self):
        def boolean(value):
            if value <= 0.5:
                return False
            elif value > 0.5:
                return True
        return [boolean(0.5*(self.s[i]+1)) for i in range(self.SATproblem.number_of_variables)]

    def check_solution(self) -> bool:
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

        test_solution = self.x()
        for i, row in enumerate(self.SATproblem.clauses):
            if not check_row(row, test_solution):
                incorrect_flag = True
                break

        if incorrect_flag:
            return False
        else:
            return True

    def solve(self):
        #while(not self.check_solution()):
        #for i in range(self.integrator.Nmax):
        new_s = self.integrator.step(self.s, lambda s : self.mgradV(s))
        new_a = self.integrator.step(self.a, lambda a, s : np.array([a[m] * self.K_m(m, s) for m in range(self.SATproblem.number_of_clauses)]) )
        self.s = new_s
        self.a = new_a
        print("step " + str(i) + '\n')


