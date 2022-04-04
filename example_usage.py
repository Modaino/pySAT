#########################################
#                                       #
# Boolean satisfiability problem solver #
#      example usage python file        #
#                                       #
# Written by Áron Vízkeleti             #
#       on 2022-04-04                   #
#       last modified 2022-04-04        #
#                                       #
#########################################

from pySAT import SAT, CTD, RHS_TYPE_THREE, ORTANT
import numpy as np

def plot_traj(sol, number_of_variables, hide_legends = False):
    import matplotlib.pyplot as plt
    plt.grid(True)
    plt.title('Spin variables')
    for i, spin_var in enumerate(sol.y[0:number_of_variables]):
        x = sol.t
        y = spin_var
        plt.plot(x, y, label='s'+str(i+1))
    if not hide_legends:
        plt.legend()
    plt.show()

def plot_aux(sol, number_of_variables, hide_legends = False):
    import matplotlib.pyplot as plt
    plt.grid(True)
    plt.title('Aux variables')
    for i, spin_var in enumerate(sol.y[number_of_variables:]):
        x = sol.t
        y = spin_var
        plt.plot(x, y, label='s'+str(i+1))
    if not hide_legends:
        plt.legend()
    plt.show()


so_file_name = 'c_libs/cSat.so'
myProblem = SAT("SAT_problems\\random3SATn15a4.266666666666667.cnf", so_file_name, rhs_type=RHS_TYPE_THREE)
N, M = myProblem.number_of_variables, myProblem.number_of_clauses
init_s =  2*np.random.rand(N) - np.ones(N)

solver = CTD(myProblem, initial_s=init_s, random_aux=False)
solver.fast_solve(t_max=50, solver_type='RK45', exit_type=ORTANT)
plot_traj(solver.sol, myProblem.number_of_variables, True)
plot_aux(solver.sol, N, True)
