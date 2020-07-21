#include "stdio.h"
#include "math.h"
union single_data {
    unsigned int u;
    int i;
    float f;
    unsigned char c[4];
} a;
union double_data {
    double d;
    unsigned char c[8];
} b;

//Out put actual bit patterns in memory
//Because computer is little-endian, print high index first.
void print4()
{
    printf("%02X %02X %02X %02X\n", a.c[3], a.c[2], a.c[1], a.c[0]);
}
void print8()
{
    printf("%02X %02X %02X %02X %02X %02X %02X %02X\n", b.c[7], b.c[6], b.c[5], b.c[4], b.c[3], b.c[2], b.c[1], b.c[0]);
}

int main(int argc, char const *argv[])
{
    //Set value -1 in int type
    a.i = -1;
    printf("Int type value -1 in memory is:\n");
    //Print bit patterns.
    print4();
    printf("Same data in memory show as unsigned int is %u\n", a.u);

    printf("Float type value 1 in memory is:\n");
    a.f = 1;
    print4();

    printf("Double type value 1.5 in memory is:\n");
    b.d = 1.5;
    print8();

    printf("1.0 / 0.0 in memory:\n");
    b.d = 1.0 / 0.0;
    print8();

    printf("-1.0 / 0.0 in memory:\n");
    b.d = -1.0 / 0.0;
    print8();

    printf("0.0 / 0.0 in memory:\n");
    b.d = 0.0 / 0.0;
    print8();

    printf("2^1000 in memory:\n");
    b.d = pow(2, 1000);
    print8();

    return 0;
}