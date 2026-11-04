#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <mpi.h>

#define REPEATS 5

int main(int argc, char **argv)
{
    int rank, size, n = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc == 2) {
            char *end;
            errno = 0;
            long value = strtol(argv[1], &end, 10);
            if (!errno && end != argv[1] && *end == '\0' &&
                value > 0 && value <= INT_MAX && value % size == 0)
                n = (int)value;
        }
        if (n == 0)
            fprintf(stderr, "Usage: mpi_array_sum N\n"
                    "N must be a positive integer divisible by %d.\n", size);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0) {
        MPI_Finalize();
        return 1;
    }

    int local_n = n / size;
    int *data = NULL;
    int *local_data = malloc((size_t)local_n * sizeof(int));
    long long *local_sums = NULL;
    if (rank == 0) {
        data = malloc((size_t)n * sizeof(int));
        local_sums = malloc((size_t)size * sizeof(long long));
    }
    if (!local_data || (rank == 0 && (!data || !local_sums))) {
        fprintf(stderr, "Rank %d: not enough memory.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    if (rank == 0)
        for (int i = 0; i < n; i++)
            data[i] = i + 1;

    long long local_sum = 0, gathered_sum = 0, reduced_sum = 0, all_sum = 0;
    long long expected = (long long)n * ((long long)n + 1) / 2;
    double total_time = 0.0;
    int correct = 1;

    for (int r = -1; r < REPEATS; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        MPI_Scatter(data, local_n, MPI_INT, local_data, local_n,
                    MPI_INT, 0, MPI_COMM_WORLD);
        local_sum = 0;
        for (int i = 0; i < local_n; i++)
            local_sum += local_data[i];

        MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, local_sums, 1,
                   MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        gathered_sum = 0;
        if (rank == 0)
            for (int i = 0; i < size; i++)
                gathered_sum += local_sums[i];

        MPI_Reduce(&local_sum, &reduced_sum, 1, MPI_LONG_LONG_INT,
                   MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Allreduce(&local_sum, &all_sum, 1, MPI_LONG_LONG_INT,
                      MPI_SUM, MPI_COMM_WORLD);

        double elapsed = MPI_Wtime() - start;
        double max_time = 0.0;
        
        MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && r >= 0)
            total_time += max_time;
        if (all_sum != expected ||
            (rank == 0 && (gathered_sum != expected || reduced_sum != expected)))
            correct = 0;
    }

    printf("rank=%d local_n=%d local_sum=%lld allreduce=%lld check=%s\n",
           rank, local_n, local_sum, all_sum, correct ? "OK" : "FAIL");
    if (rank == 0)
        printf("N=%d processes=%d expected=%lld gather=%lld reduce=%lld "
               "allreduce=%lld time_s=%.9f repeats=%d check=%s\n",
               n, size, expected, gathered_sum, reduced_sum, all_sum,
               total_time / REPEATS, REPEATS, correct ? "OK" : "FAIL");

    free(data);
    free(local_data);
    free(local_sums);
    MPI_Finalize();
    return correct ? 0 : 1;
}
