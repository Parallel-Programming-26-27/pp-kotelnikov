#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

#define REPEATS 5
#define TOL 1e-8

/* mode: 0 = serial, 1 = OpenMP, 2 = SIMD, 3 = OpenMP + SIMD.*/
void train(const double *x, const double *y, double *w, double *b,
           double *g, double *partial, int n, int d, int iterations,
           int threads, double lambda, double eta, int mode)
{
    int stride = d + 8;
    for (int k = 0; k < iterations; k++) {
        memset(g, 0, (size_t)d * sizeof(double));
        double gb = 0.0;

        if (mode == 0) {
            for (int i = 0; i < n; i++) {
                const double *row = x + (size_t)i * d;
                double prediction = *b;
                for (int j = 0; j < d; j++)
                    prediction += row[j] * w[j];
                double error = prediction - y[i];
                gb += error;
                for (int j = 0; j < d; j++)
                    g[j] += error * row[j];
            }
        } else if (mode == 2) {
            for (int i = 0; i < n; i++) {
                const double *row = x + (size_t)i * d;
                double prediction = 0.0;
#pragma omp simd reduction(+:prediction)
                for (int j = 0; j < d; j++)
                    prediction += row[j] * w[j];
                double error = prediction + *b - y[i];
                gb += error;
#pragma omp simd
                for (int j = 0; j < d; j++)
                    g[j] += error * row[j];
            }
        } else {
            memset(partial, 0, (size_t)threads * stride * sizeof(double));
#pragma omp parallel num_threads(threads) reduction(+:gb)
            {
                double *local = partial + (size_t)omp_get_thread_num() * stride;
#pragma omp for schedule(static)
                for (int i = 0; i < n; i++) {
                    const double *row = x + (size_t)i * d;
                    double prediction = 0.0;
                    if (mode == 1) {
                        for (int j = 0; j < d; j++)
                            prediction += row[j] * w[j];
                        double error = prediction + *b - y[i];
                        gb += error;
                        for (int j = 0; j < d; j++)
                            local[j] += error * row[j];
                    } else {
#pragma omp simd reduction(+:prediction)
                        for (int j = 0; j < d; j++)
                            prediction += row[j] * w[j];
                        double error = prediction + *b - y[i];
                        gb += error;
#pragma omp simd
                        for (int j = 0; j < d; j++)
                            local[j] += error * row[j];
                    }
                }
            }
            for (int t = 0; t < threads; t++)
                for (int j = 0; j < d; j++)
                    g[j] += partial[(size_t)t * stride + j];
        }

        for (int j = 0; j < d; j++) {
            double z = w[j] - eta * g[j] / n;
            double threshold = eta * lambda;
            if (z > threshold)
                w[j] = z - threshold;
            else if (z < -threshold)
                w[j] = z + threshold;
            else
                w[j] = 0.0;
        }
        *b -= eta * gb / n; 
    }
}

double loss(const double *x, const double *y, const double *w, double b,
            int n, int d, double lambda)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double prediction = b;
        for (int j = 0; j < d; j++)
            prediction += x[(size_t)i * d + j] * w[j];
        double error = prediction - y[i];
        sum += error * error;
    }
    sum /= 2.0 * n;
    for (int j = 0; j < d; j++)
        sum += lambda * fabs(w[j]);
    return sum;
}

int main(int argc, char **argv)
{
    int n = 100000, d = 32, iterations = 60, threads = 8;
    double lambda = 0.01, eta = 0.5;
    if (argc != 1 && argc != 7) {
        printf("usage: %s [N d K threads lambda eta]\n", argv[0]);
        return 1;
    }
    if (argc == 7) {
        int *parameters[] = {&n, &d, &iterations, &threads};
        for (int i = 0; i < 4; i++) {
            char *end;
            long value = strtol(argv[i + 1], &end, 10);
            if (*end || value <= 0 || value > INT_MAX - 8) {
                fprintf(stderr, "N, d, K and threads must be positive integers.\n");
                return 1;
            }
            *parameters[i] = (int)value;
        }
        char *end;
        lambda = strtod(argv[5], &end);
        if (end == argv[5] || *end || !isfinite(lambda) || lambda < 0) {
            fprintf(stderr, "lambda must be finite and nonnegative.\n");
            return 1;
        }
        eta = strtod(argv[6], &end);
        if (end == argv[6] || *end || !isfinite(eta) || eta <= 0) {
            fprintf(stderr, "eta must be finite and positive.\n");
            return 1;
        }
    }
    if ((size_t)n > (size_t)-1 / sizeof(double) / d ||
        (size_t)threads > (size_t)-1 / sizeof(double) / (d + 8)) {
        fprintf(stderr, "Array size is too large.\n");
        return 1;
    }

    double *x = malloc((size_t)n * d * sizeof(double));
    double *y = malloc((size_t)n * sizeof(double));
    double *w = calloc((size_t)d, sizeof(double));
    double *reference = calloc((size_t)d, sizeof(double));
    double *g = malloc((size_t)d * sizeof(double));
    double *partial = malloc((size_t)threads * (d + 8) * sizeof(double));
    if (!x || !y || !w || !reference || !g || !partial) {
        fprintf(stderr, "Not enough memory.\n");
        free(x); free(y); free(w); free(reference); free(g); free(partial);
        return 1;
    }

    srand(42);
    for (int i = 0; i < n; i++) {
        y[i] = 0.5;
        for (int j = 0; j < d; j++) {
            double value = 2.0 * rand() / RAND_MAX - 1.0;
            x[(size_t)i * d + j] = value;
            if (j % 8 == 0)
                y[i] += (j % 16 == 0 ? 1.0 : -1.0) * value;
        }
        y[i] += 0.02 * (2.0 * rand() / RAND_MAX - 1.0);
    }

    omp_set_dynamic(0);
    const char *names[] = {"serial", "openmp", "simd", "openmp_simd"};
    double serial_time = 0.0, reference_loss = 0.0, reference_b = 0.0;
    double initial_loss = loss(x, y, w, 0.0, n, d, lambda);
    int success = 1;
    printf("N=%d d=%d K=%d threads=%d lambda=%g eta=%g seed=42 repeats=%d tol=%g\n",
           n, d, iterations, threads, lambda, eta, REPEATS, TOL);
    printf("initial_loss=%.12g processors=%d max_threads=%d\n",
           initial_loss, omp_get_num_procs(), omp_get_max_threads());
    printf("variant,threads,loss,max_dw,abs_db,abs_dloss,zeros,time_s,speedup,check\n");

    for (int mode = 0; mode < 4; mode++) {
        double times[REPEATS], b = 0.0;
        for (int r = -1; r < REPEATS; r++) {
            memset(w, 0, (size_t)d * sizeof(double));
            b = 0.0;
            double start = omp_get_wtime();
            train(x, y, w, &b, g, partial, n, d, iterations,
                  threads, lambda, eta, mode);
            double elapsed = omp_get_wtime() - start;
            if (r >= 0)
                times[r] = elapsed;
        }

        for (int i = 0; i < REPEATS; i++)
            for (int j = i + 1; j < REPEATS; j++)
                if (times[j] < times[i]) {
                    double temp = times[i]; times[i] = times[j]; times[j] = temp;
                }
        double value = loss(x, y, w, b, n, d, lambda);
        double dw = 0.0;
        int zeros = 0, finite = isfinite(value) && isfinite(b);
        if (mode == 0) {
            memcpy(reference, w, (size_t)d * sizeof(double));
            reference_loss = value;
            reference_b = b;
            serial_time = times[REPEATS / 2];
        }
        for (int j = 0; j < d; j++) {
            if (!isfinite(w[j])) finite = 0;
            if (fabs(w[j] - reference[j]) > dw)
                dw = fabs(w[j] - reference[j]);
            if (fabs(w[j]) < 1e-6) zeros++;
        }
        double db = fabs(b - reference_b), df = fabs(value - reference_loss);
        int ok = finite && dw <= TOL && db <= TOL && df <= TOL;
        if (!ok) success = 0;
        double median = times[REPEATS / 2];
        printf("%s,%d,%.12g,%.3e,%.3e,%.3e,%d,%.6f,%.3f,%s\n",
               names[mode], mode == 0 || mode == 2 ? 1 : threads,
               value, dw, db, df, zeros, median,
               median > 0 && serial_time > 0 ? serial_time / median : NAN, ok ? "OK" : "FAIL");
    }

    free(x); free(y); free(w); free(reference); free(g); free(partial);
    return success ? 0 : 1;
}
