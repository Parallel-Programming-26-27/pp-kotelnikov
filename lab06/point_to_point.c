#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define TAG_MESSAGE 10
#define TAG_REPLY 20

int main(int argc, char **argv)
{
    int rank, size;
    char message[128], reply[128];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "Run with at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            snprintf(message, sizeof(message), "Hello to rank %d from rank 0", r);
            printf("Rank 0 sends to rank %d: %s\n", r, message);
            fflush(stdout);
            MPI_Send(message, (int)strlen(message) + 1, MPI_CHAR,
                     r, TAG_MESSAGE, MPI_COMM_WORLD);
        }

        /*ответы принимаются рангов с отдельным тегом*/
        int replies = 0;
        for (int r = 1; r < size; r++) {
            MPI_Recv(reply, sizeof(reply), MPI_CHAR,
                     r, TAG_REPLY, MPI_COMM_WORLD, &status);
            printf("Rank 0 received from rank %d: %s\n", status.MPI_SOURCE, reply);
            replies++;
        }
        printf("Rank 0: sent %d messages, received %d replies\n", size - 1, replies);
    } else {
        /* сначала принимаем сообщение а потом отвечаем*/
        MPI_Recv(message, sizeof(message), MPI_CHAR,
                 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
        printf("Rank %d received from rank %d: %s\n", rank, status.MPI_SOURCE, message);

        snprintf(reply, sizeof(reply), "Message received by rank %d", rank);
        printf("Rank %d sends reply to rank 0\n", rank);
        fflush(stdout);
        MPI_Send(reply, (int)strlen(reply) + 1, MPI_CHAR,
                 0, TAG_REPLY, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
