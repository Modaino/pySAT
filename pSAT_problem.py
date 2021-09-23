#Boolean satisfiability problem solver with continuous dynamical system
import numpy as np

class SATProblem:
    def __init__(self, cnf_file_name):
        with open(cnf_file_name) as cnf_file:
            lines = cnf_file.readlines()
            self.number_of_variables = lines[0].split(' ')[2]
            self.number_of_clauses = lines[0].split(' ')[3]

            #(v)^(v)#
            clause_and = []
            for i in range(int(self.number_of_clauses) + 1):
                if i > 0:
                    clause_or = []
                    for variable_str in lines[i].split(' '):
                        variable = int(variable_str)
                        if variable != 0:
                            clause_or.append(variable)
                    clause_and.append(clause_or)

            self.clauses = clause_and

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

class RK5:
    def __init__(self, dim, f, df = None, y_init = None, Nmax = 10000) -> None:
        if not dim or not f:
            raise ValueError
        
        if not y_init:
            self.y_init = np.random.randn(*(dim, 1))
        else:
            self.y_init = y_init

    def step(f, df, h, y_n):
        k1 = h * f(y_n)
        k2 = h * f(y_n + k1*1/3 + h * np.dot(df(y_n), k1))
        k3 = h * f(y_n + k1*152/125 + k2*252/125 - h * 44/125 * np.dot(df(y_n), k1))
        k4 = h * f(y_n + k1*19/2 - k2*72/7 + k3*25/14 + h * 5/2 * np.dot(df(y_n), k1))
        return y_n + 5/48*k1 + 27/56*k2 + 125/336*k3 + 1/24*k4

class CTDSolver:
    def __init__(self, SATproblem) -> None:
        self.SATproblem = SATproblem