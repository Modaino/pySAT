#Python script to verify a SAT solution to a k-SAT problem

#cnf_file_name = 'test_problem.cnf'
#sol_file_name = 'test_sol.out'
from sys import float_repr_style

assignment = 'assignment_T24'
cnf_file_name = 'E:\Aron\BMW_SAT\\bmw_problems\\' + assignment + '.cnf'
#sol_file_name = 'E:\Aron\AnalogSat_\Frontend\\bmw_solutions\\solution_T24.out'
sol_file_name = 'E:\Aron\BMW_SAT\\bmw_sol\\solution_T24_r.out'

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


with open(cnf_file_name) as cnf_file, open(sol_file_name) as sol_file:
    lines = cnf_file.readlines()
    number_of_variables = lines[0].split(' ')[2]
    number_of_lines = lines[0].split(' ')[3]

    lines2 = sol_file.readlines()
    solution = []
    for element_str in lines2[1].split(' '):
        element = int(element_str)
        if element != 0:
            if element > 0:
                solution.append(True)
            elif element < 0:
                solution.append(False)

    #(v)^(v)#
    clause_and = []
    for i in range(int(number_of_lines) + 1):
        if i > 0:
            clause_or = []
            for variable_str in lines[i].split(' '):
                variable = int(variable_str)
                if variable != 0:
                    clause_or.append(variable)
            clause_and.append(clause_or)

    incorrect_flag = False
    for i, row in enumerate(clause_and):
        if not check_row(row, solution):
            incorrect_flag = True
            print(row)
            print(i)
            print([solution[abs(x)] for x in row])
            raise RuntimeError
            break

    if incorrect_flag:
        print('The solution "'+ sol_file_name +'" to '+ assignment +' is NOT correct')
    else:
        print('The solution "'+ sol_file_name +'" to '+ assignment +' is correct')
    