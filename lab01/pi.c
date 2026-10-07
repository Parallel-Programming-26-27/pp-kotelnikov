#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEFAULT_TERMS 1000000000LL

typedef struct {
    double pi;
    double seconds;
    int threads;
} pi_result;

static int parse_positive(const char *arg, long long max, long long *out) {
    char *end = NULL;
    errno = 0;
    const long long value = strtoll(arg, &end, 10);
    if (errno != 0 || end == arg || *end != '\0' || value <= 0 || value > max) {
        return 0;
    }
    *out = value;
    return 1;
}

static pi_result pi_sequential(long long n) {
    const double start = omp_get_wtime();

    double sum = 0.0;
    for (long long i = 0; i < n; ++i) {
        const double term = 1.0 / (2.0 * (double) i + 1.0);
        sum += (i & 1LL) ? -term : term;
    }

    const pi_result result = {
        .pi = 4.0 * sum,
        .seconds = omp_get_wtime() - start,
        .threads = 1
    };
    return result;
}

static pi_result pi_parallel(long long n, int threads) {
    omp_set_dynamic(0);
    const double start = omp_get_wtime();

    double sum = 0.0;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:sum)
    for (long long i = 0; i < n; ++i) {
        const double term = 1.0 / (2.0 * (double) i + 1.0);
        sum += (i & 1LL) ? -term : term;
    }

    const pi_result result = {
        .pi = 4.0 * sum,
        .seconds = omp_get_wtime() - start,
        .threads = threads
    };
    return result;
}

static void print_result(const char *label, long long n, pi_result r) {
    printf("[%s]\n", label);
    printf("  n          = %lld\n", n);
    printf("  threads    = %d\n", r.threads);
    printf("  pi         = %.15f\n", r.pi);
    printf("  abs error  = %.3e\n", fabs(r.pi - M_PI));
    printf("  time, s    = %.6f\n", r.seconds);
}

int main(int argc, char **argv) {
    long long n = DEFAULT_TERMS;
    long long threads = omp_get_max_threads();
    if (argc > 3
        || (argc > 1 && !parse_positive(argv[1], LLONG_MAX, &n))
        || (argc > 2 && !parse_positive(argv[2], INT_MAX, &threads))) {
        fprintf(stderr, "usage: %s [n > 0] [threads > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const pi_result seq = pi_sequential(n);
    const pi_result par = pi_parallel(n, (int) threads);

    print_result("sequential", n, seq);
    print_result("parallel", n, par);
    printf("speedup      = %.2f\n", seq.seconds / par.seconds);
    return EXIT_SUCCESS;
}
