#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define PI 3.14159265358979323846
#define A 0.0
#define B (PI / 2)
#define EXACT (PI / 4)

double f(double x)
{
    double c = cos(6 * x);
    return c * c;
}

double integral_seq(double a, double b, long long n)
{
    double h = (b - a) / n;
    double sum = 0.0;
    for (long long i = 0; i < n; i++)
        sum += f(a + (i + 0.5) * h);
    return sum * h;
}

double integral_par(double a, double b, long long n, int threads)
{
    double h = (b - a) / n;
    double sum = 0.0;
#pragma omp parallel for num_threads(threads) reduction(+:sum)
    for (long long i = 0; i < n; i++)
        sum += f(a + (i + 0.5) * h);
    return sum * h;
}

double integral(double a, double b, long long n, int threads)
{
    if (threads == 1)
        return integral_seq(a, b, n);
    return integral_par(a, b, n, threads);
}

double integral_eps(double a, double b, double eps, int threads, long long *n)
{
    *n = 1;
    double prev = integral(a, b, *n, threads);
    double cur = integral(a, b, 2 * *n, threads);
    while (fabs(cur - prev) >= eps) {
        *n *= 2;
        prev = cur;
        cur = integral(a, b, 2 * *n, threads);
    }
    *n *= 2;
    return cur;
}

void run(double a, double b, long long n, int threads)
{
    double t = omp_get_wtime();
    double value = integral(a, b, n, threads);
    t = omp_get_wtime() - t;
    printf("integral = %.15f\n", value);
    printf("time     = %.6f s\n", t);
    printf("threads  = %d\n", threads);
}

void run_eps(double a, double b, double eps, int threads)
{
    long long n;
    double t = omp_get_wtime();
    double value = integral_eps(a, b, eps, threads, &n);
    t = omp_get_wtime() - t;
    printf("integral = %.15f\n", value);
    printf("n        = %lld\n", n);
    printf("time     = %.6f s\n", t);
    printf("threads  = %d\n", threads);
}

void experiment1(void)
{
    long long n = 100000000;
    int threads[] = {1, 2, 4, 8, 16};

    double t = omp_get_wtime();
    double value = integral_seq(A, B, n);
    double t_seq = omp_get_wtime() - t;

    printf("Experiment 1: n = %lld\n", n);
    printf("sequential: integral = %.15f, time = %.6f s\n", value, t_seq);
    printf("threads\tintegral\t\ttime, s\t\tspeedup\n");
    for (int k = 0; k < 5; k++) {
        t = omp_get_wtime();
        value = integral_par(A, B, n, threads[k]);
        t = omp_get_wtime() - t;
        printf("%d\t%.15f\t%.6f\t%.2f\n", threads[k], value, t, t_seq / t);
    }
}

void experiment2(void)
{
    printf("Experiment 2: threads = 8\n");
    printf("n\t\tintegral\t\terror\t\ttime, s\n");
    for (long long n = 10000; n <= 100000000; n *= 10) {
        long long repeats = 100000000 / n;
        double value = 0.0;
        double t = omp_get_wtime();
        for (long long r = 0; r < repeats; r++)
            value = integral_par(A, B, n, 8);
        t = (omp_get_wtime() - t) / repeats;
        printf("%lld\t%.15f\t%.2e\t%.8f\n", n, value, fabs(value - EXACT), t);
    }
}

void experiment3(double eps)
{
    int repeats = 10000;
    int threads[] = {1, 8};
    long long n;
    printf("Experiment 3: eps = %g\n", eps);
    printf("threads\tn\tintegral\t\ttime, s\n");
    for (int k = 0; k < 2; k++) {
        volatile double value = 0.0;
        double t = omp_get_wtime();
        for (int r = 0; r < repeats; r++)
            value = integral_eps(A, B, eps, threads[k], &n);
        t = (omp_get_wtime() - t) / repeats;
        printf("%d\t%lld\t%.15f\t%.8f\n", threads[k], n, value, t);
    }
}

int main(int argc, char **argv)
{
    if (argc == 6 && strcmp(argv[1], "run") == 0) {
        run(atof(argv[2]), atof(argv[3]), atoll(argv[4]), atoi(argv[5]));
    } else if (argc == 6 && strcmp(argv[1], "eps") == 0) {
        run_eps(atof(argv[2]), atof(argv[3]), atof(argv[4]), atoi(argv[5]));
    } else if (argc == 2 && strcmp(argv[1], "exp1") == 0) {
        experiment1();
    } else if (argc == 2 && strcmp(argv[1], "exp2") == 0) {
        experiment2();
    } else if (argc == 3 && strcmp(argv[1], "exp3") == 0) {
        experiment3(atof(argv[2]));
    } else {
        printf("usage:\n");
        printf("  %s run a b n threads\n", argv[0]);
        printf("  %s eps a b eps threads\n", argv[0]);
        printf("  %s exp1 | exp2 | exp3 eps\n", argv[0]);
        return 1;
    }
    return 0;
}
