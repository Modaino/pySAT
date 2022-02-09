/*                                                *
 *      C\C++ library for CTDS simulations        *
 *                                                *
 *     Written by Aron Vizkeleti on 2021-12-10    *
 *           last modified 2022-02-03             *
 *                                                *
 *  to compile use:                               *
 *  cc -std=c99 -fPIC -shared -o cSAT.so cSAT.c   *
 *                                                */


//Helper functions (not to be called from outside)

double k_mi(int m, int i, double s[], int c[], int number_of_variables){
    double productum = 1.0;
    for (int j = 0; j < number_of_variables; j++)
    {   
        if (i != j){
            productum *= (1 - c[m*number_of_variables + j] * s[j]);
        }
    }    
    return 0.125 * productum; // only for 3SAT 0.125 = 2^-3
}

double k_mi_K_m(int m, int i, double s[], int c[], int number_of_variables){
    double a = k_mi(m, i, s, c, number_of_variables);
    return a*a*(1 - c[m*number_of_variables + i] * s[i]);
}

double K_m(int m, double s[], int c[], int number_of_variables){
    double productum = 1.0;
    for (int j = 0; j < number_of_variables; j++)
    {   
            productum *= (1 - c[m*number_of_variables + j] * s[j]);
    }    
    return 0.125 * productum; // only for 3SAT 0.125 = 2^-3
}

double gradV_i(int i, double s[], double a[], int c[], int number_of_variables, int number_of_clauses){
    double summ = 0.0;
    for (int m = 0; m < number_of_clauses; m++)
    {
        summ += 2*a[m]*c[m*number_of_variables + i]*k_mi_K_m(m, i, s, c, number_of_variables);
    }
    return summ;
}

double K_m_squared(int m, double s[], int c[], int number_of_variables){
    double productum = 1.0;
    for (int j = 0; j < number_of_variables; j++)
    {   
            productum *= (1 - c[m*number_of_variables + j] * s[j]);
    }    
    return 0.125 * 0.125 * productum*productum;
}

//To be called from python

void rhs1(int N, int M, int c[], double y[], double result[]){
    double *s = y;
    double *a = y + N;
    for (int i = 0; i < N; i++)
    {
        result[i] = gradV_i(i, s, a, c, N, M);
    }
    for (int i = N; i < N+M; i++)
    {
        result[i] = y[i] * K_m(i-N, s, c, N);
    }
}

void rhs2(int N, int M, int c[], double y[], double result[]){
    double *s = y;
    double *a = y + N;
    for (int i = 0; i < N; i++)
    {
        result[i] = gradV_i(i, s, a, c, N, M);
    }
    for (int i = N; i < N+M; i++)
    {
        result[i] = y[i] * K_m_squared(i-N, s, c, N);
    }
}

void jacobian1(int N, int M, int c[], double y[], double result[]){
    double *s = y;
    double *a = y + N;
    //J11
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double kronecker_ij = 0.0;
            if (i == j) { kronecker_ij = 1.0; }
            double summ = 0;
            for (int m = 0; m < M; m++)
            {
                double prod = 1;
                for (int l = 0; l < N; l++)
                {
                    if (l != i && l != j)
                    {
                        prod *= (1-c[m*N + l]*s[l]);
                    }
                }
                //2*2^(-2k); for SAT3 only!!!
                summ += 0.03125*a[m]*c[m*N + i]*(-c[m*N + j])*(1-c[m*N + j]*s[j])*(1-c[m*N + i]*s[i]) * (1+kronecker_ij) * prod;
            }
            
            result[i*(N+M) + j] = summ;
        }
    }
    //J12
    for (int i = 0; i < N; i++)
    {
        for (int j = N; j < N+M; j++)
        {
            result[i*(N+M) + j] = 2*c[j*N + i] * k_mi_K_m(j, i, s, c, N);
        }
    }
    //J21
    for (int i = N; i < N+M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i*(N+M) + j] = a[i]*(-c[i*N + j])*k_mi(i, j, s, c, N);
        }
    }
    //J22
    for (int i = N; i < N+M; i++)
    {
        for (int j = N; j < N+M; j++)
        {
            if (i == j) {
                result[i*(N+M) + j] = K_m(i, s, c, N);
            }
            else {
                result[i*(N+M) + j] = 0.0;
            }
        }
    }
}

void jacobian2(int N, int M, int c[], double y[], double result[]){
    //NOT YET IMPLEMENTED!!!!
    double *s = y;
    double *a = y + N;
    //J11
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double kronecker_ij = 0.0;
            if (i == j) { kronecker_ij = 1.0; }
            double summ = 0;
            for (int m = 0; m < M; m++)
            {
                double prod = 1;
                for (int l = 0; l < N; l++)
                {
                    if (l != i && l != j)
                    {
                        prod *= (1-c[m*N + l]*s[l]);
                    }
                }
                //2*2^(-2k); for SAT3 only!!!
                summ += 0.03125*a[m]*c[m*N + i]*(-c[m*N + j])*(1-c[m*N + j]*s[j])*(1-c[m*N + i]*s[i]) * (1+kronecker_ij) * prod;
            }
            
            result[i*(N+M) + j] = summ;
        }
    }
    //J12
    for (int i = 0; i < N; i++)
    {
        for (int j = N; j < N+M; j++)
        {
            result[i*(N+M) + j] = 2*c[j*N + i] * k_mi_K_m(j, i, s, c, N);
        }
    }
    //J21
    for (int i = N; i < N+M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i*(N+M) + j] = a[i]*(-c[i*N + j])*k_mi(i, j, s, c, N);
        }
    }
    //J22
    for (int i = N; i < N+M; i++)
    {
        for (int j = N; j < N+M; j++)
        {
            if (i == j) {
                result[i*(N+M) + j] = K_m(i, s, c, N);
            }
            else {
                result[i*(N+M) + j] = 0.0;
            }
        }
    }
}