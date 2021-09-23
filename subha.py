#Python file to check SAT_verify.py

with open('SAT_Generator\\100.csv', 'r') as subha_file, open('SAT_Generator\\100.cnf', 'w') as prb_file, open('SAT_Generator\\100.out', 'w') as sol_file:
    line1 = subha_file.readline()
    clauses = line1.split('|')
    header = 'p cnf 30 ' + str(len(clauses) - 1) + '\n'

    clauses_to_write = [clause + '0\n' for clause in clauses]
    clauses_to_write.pop()
    prb_file.writelines([header])
    prb_file.writelines(clauses_to_write)

    line2 = subha_file.readline()
    solution = (line2.replace('{', '').replace('}', '').replace(' ', '').replace('\n', '').split(','))
    sol_lines = ['SAT\n']
    for i, variable in enumerate(solution):
        if int(variable) == 0:
            sol_lines.append(str(-(i + 1)) + ' ')
        elif int(variable) == 1:
            sol_lines.append(str((i + 1)) + ' ')
    #sol_lines.append('\n')
    sol_file.writelines(sol_lines)
