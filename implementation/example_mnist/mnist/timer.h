#include <cstddef>
#include <sys/time.h>

// timer
double get_wtime(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6; // seconds
}