#include "stdio.h"
int main(void)
{
    int s = 0;
    int i;

    for (i = 1; i < 8; i++)
    {
        int j = 1;
        while (i >= j)
        {
            s = j + s;
            j = j + 1;
        }
    }
    printf("%d", s);
    return 0;
}