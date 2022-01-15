double K_mi(int m, int i, double s[], int c[], int number_of_variables){
    double productum = 1.0;
    for (int j = 0; j < number_of_variables; j++)
    {   
        if (i != j){
            productum *= (1 - c[m*number_of_variables + j] * s[j]);
        }
    }    
    return 0.125 * productum; // only for 3SAT 0.125 = 2^-3
}

double K_mi_m(int m, int i, double s[], int c[], int number_of_variables){
    double a = K_mi(m, i, s, c, number_of_variables);
    return a*a*(1 - c[m*number_of_variables + i] * s[i]);
}

double K_m(int m, double s[], int c[], int number_of_variables){
    double productum = 1.0;
    for (int j = 0; j < number_of_variables; j++)
    {   
            productum *= (1 - c[m*number_of_variables + j] * s[j]);
    }    
    return productum;
}

double gradV_i(int i, double s[], double a[], int c[], int number_of_variables, int number_of_clauses){
    double summ = 0.0;
    for (int m = 0; m < number_of_clauses; m++)
    {
        summ += 2*a[m]*c[m*number_of_variables + i]*K_mi_m(m, i, s, c, number_of_variables);
    }
    return summ;
}
