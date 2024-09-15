#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

/*
    USAGE  : compile with -lm (and why not -O3)
            redirect the standard output to a text file
            mpicc heatsink.c -O3 -lm -o heatsink
            mpiexec ./heatsink <partition_choice: from 1 to 3> > fichier4.txt
            then run the indicated python script for graphical rendering

            partition_choice:
            1: 1D
            2: 2D
            3: 3D

    DISCLAIMER : this code does not claim to an absolute realism.
                this code could be obviously improved, but has been written so as
    			to make as clear as possible the physics principle of the simulation.

*/

/* one can change the matter of the heatsink, its size, the power of the CPU, etc. */
#define ALUMINIUM
#define FAST              /* MEDIUM is faster, and FAST is even faster (for debugging) */
#define DUMP_STEADY_STATE

const double L = 0.15;      /* length (x) of the heatsink (m) */
const double l = 0.12;      /* height (y) of the heatsink (m) */
const double E = 0.008;     /* width (z) of the heatsink (m) */
const double watercooling_T = 20;   /* temperature of the fluid for water-cooling, (°C) */
const double CPU_TDP = 280; /* power dissipated by the CPU (W) */

/* dl: "spatial step" for simulation (m) */
/* dt: "time step" for simulation (s) */
#ifdef FAST
double dl = 0.004;
double dt = 0.004;
#endif

#ifdef MEDIUM
double dl = 0.002;
double dt = 0.002;
#endif

#ifdef NORMAL
double dl = 0.001;
double dt = 0.001;
#endif

#ifdef CHALLENGE
double dl = 0.0001;
double dt = 0.00001;
#endif

/* sink_heat_capacity: specific heat capacity of the heatsink (J / kg / K) */
/* sink_density: density of the heatsink (kg / m^3) */
/* sink_conductivity: thermal conductivity of the heatsink (W / m / K) */
/* euros_per_kg: price of the matter by kilogram */
#ifdef ALUMINIUM
double sink_heat_capacity = 897;
double sink_density = 2710;
double sink_conductivity = 237;
double euros_per_kg = 1.594;
#endif

#ifdef COPPER
double sink_heat_capacity = 385;
double sink_density = 8960;
double sink_conductivity = 390;
double euros_per_kg = 5.469;
#endif

#ifdef GOLD
double sink_heat_capacity = 128;
double sink_density = 19300;
double sink_conductivity = 317;
double euros_per_kg = 47000;
#endif

#ifdef IRON
double sink_heat_capacity = 444;
double sink_density = 7860;
double sink_conductivity = 80;
double euros_per_kg = 0.083;
#endif

const double Stefan_Boltzmann = 5.6703e-8;  /* (W / m^2 / K^4), radiation of black body */
const double heat_transfer_coefficient = 10;    /* coefficient of thermal convection (W / m^2 / K) */
double CPU_surface;

/*
 * Return True if the CPU is in contact with the heatsink at the point (x,y).
 * This describes an AMD EPYC "Rome".
 */
static inline bool CPU_shape(double x, double y)
{
    x -= (L - 0.0754) / 2;
    y -= (l - 0.0585) / 2;
    bool small_y_ok = (y > 0.015 && y < 0.025) || (y > 0.0337 && y < 0.0437);
    bool small_x_ok = (x > 0.0113 && x < 0.0186) || (x > 0.0193 && x < 0.0266)
        || (x > 0.0485 && x < 0.0558) || (x > 0.0566 && x < 0.0639);
    bool big_ok = (x > 0.03 && x < 0.045 && y > 0.0155 && y < 0.0435);
    return big_ok || (small_x_ok && small_y_ok);
}

/* returns the total area of the surface of contact between CPU and heatsink (in m^2) */
double CPU_contact_surface()
{
    double S = 0;
    for (double x = dl / 2; x < L; x += dl)
        for (double y = dl / 2; y < l; y += dl)
            if (CPU_shape(x, y))
                S += dl * dl;
    return S;
}

/* Returns the new temperature of the cell (i, j, k). For this, there is an access to neighbor
 * cells (left, right, top, bottom, front, back), except if (i, j, k) is on the external surface. */
static inline double update_temperature(const double *T, int u, int n, int m, int o, int i, int j, int k)
{
		/* quantity of thermal energy that must be brought to a cell to make it heat up by 1°C */
    const double cell_heat_capacity = sink_heat_capacity * sink_density * dl * dl * dl; /* J.K */
    const double dl2 = dl * dl;
    double thermal_flux = 0;

    if (i > 0)
        thermal_flux += (T[u - 1] - T[u]) * sink_conductivity * dl; /* neighbor x-1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (i < n - 1)
        thermal_flux += (T[u + 1] - T[u]) * sink_conductivity * dl; /* neighbor x+1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (j > 0)
        thermal_flux += (T[u - n] - T[u]) * sink_conductivity * dl; /* neighbor y-1 */
    else {
        /* Bottom cell: does it receive it from the CPU ? */
        if (CPU_shape(i * dl, k * dl))
            thermal_flux += CPU_TDP / CPU_surface * dl2;
        else {
            thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
            thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
        }
    }

    if (j < m - 1)
        thermal_flux += (T[u + n] - T[u]) * sink_conductivity * dl; /* neighbor y+1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (k > 0)
        thermal_flux += (T[u - n * m] - T[u]) * sink_conductivity * dl; /* neighbor z-1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (k < o - 1)
        thermal_flux += (T[u + n * m] - T[u]) * sink_conductivity * dl; /* neighbor z+1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    /* adjust temperature depending on the heat flux */
    return T[u] + thermal_flux * dt / cell_heat_capacity;
}


/* Returns the new temperature of the cell (i, j, k). For this, there is an access to neighbor
 * cells (left, right, top, bottom, front, back), except if (i, j, k) is on the external surface. */
static inline double update_temperature2D(const double *T, int u, int n, int m, int o, int i, int j, int k, int hauteurTotal)
{
		/* quantity of thermal energy that must be brought to a cell to make it heat up by 1°C */
    const double cell_heat_capacity = sink_heat_capacity * sink_density * dl * dl * dl; /* J.K */
    const double dl2 = dl * dl;
    double thermal_flux = 0;

    if (i > 0)
        thermal_flux += (T[u - 1] - T[u]) * sink_conductivity * dl; /* neighbor x-1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (i < n - 1)
        thermal_flux += (T[u + 1] - T[u]) * sink_conductivity * dl; /* neighbor x+1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (j > 0)
        thermal_flux += (T[u - n] - T[u]) * sink_conductivity * dl; /* neighbor y-1 */
    else {
        /* Bottom cell: does it receive it from the CPU ? */
        if (CPU_shape(i * dl, k * dl))
            thermal_flux += CPU_TDP / CPU_surface * dl2;
        else {
            thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
            thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
        }
    }

    if (j < m - 1)
        thermal_flux += (T[u + n] - T[u]) * sink_conductivity * dl; /* neighbor y+1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (k > 0)
        thermal_flux += (T[u - n * hauteurTotal] - T[u]) * sink_conductivity * dl; /* neighbor z-1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (k < o - 1)
        thermal_flux += (T[u + n * hauteurTotal] - T[u]) * sink_conductivity * dl; /* neighbor z+1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    /* adjust temperature depending on the heat flux */
    return T[u] + thermal_flux * dt / cell_heat_capacity;
}

/* Returns the new temperature of the cell (i, j, k). For this, there is an access to neighbor
 * cells (left, right, top, bottom, front, back), except if (i, j, k) is on the external surface. */
static inline double update_temperature3D(const double *T, int u, int n, int m, int o, int i, int j, int k, int hauteurTotal, int longueurTotal)
{
		/* quantity of thermal energy that must be brought to a cell to make it heat up by 1°C */
    const double cell_heat_capacity = sink_heat_capacity * sink_density * dl * dl * dl; /* J.K */
    const double dl2 = dl * dl;
    double thermal_flux = 0;

    if (i > 0)
        thermal_flux += (T[u - 1] - T[u]) * sink_conductivity * dl; /* neighbor x-1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (i < n - 1)
        thermal_flux += (T[u + 1] - T[u]) * sink_conductivity * dl; /* neighbor x+1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (j > 0)
        thermal_flux += (T[u - longueurTotal] - T[u]) * sink_conductivity * dl; /* neighbor y-1 */
    else {
        /* Bottom cell: does it receive it from the CPU ? */
        if (CPU_shape(i * dl, k * dl))
            thermal_flux += CPU_TDP / CPU_surface * dl2;
        else {
            thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
            thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
        }
    }

    if (j < m - 1)
        thermal_flux += (T[u + longueurTotal] - T[u]) * sink_conductivity * dl; /* neighbor y+1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (k > 0)
        thermal_flux += (T[u - longueurTotal * hauteurTotal] - T[u]) * sink_conductivity * dl; /* neighbor z-1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (k < o - 1)
        thermal_flux += (T[u + longueurTotal * hauteurTotal] - T[u]) * sink_conductivity * dl; /* neighbor z+1 */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    /* adjust temperature depending on the heat flux */
    return T[u] + thermal_flux * dt / cell_heat_capacity;
}

double my_gettimeofday()
{
	struct timeval tmp_time;
	gettimeofday(&tmp_time, NULL);
	return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

/* Run the simulation on the k-th xy plane.
 * v is the index of the start of the k-th xy plane in the arrays T and R. */
static inline void do_xy_plane(const double *T, double *R, int v, int n, int m, int o, int k)
{
    if (k == 0)
				// we do not modify the z = 0 plane: it is maintained at constant temperature via water-cooling
        return;
    for (int j = 0; j < m; j++) {   // y
        for (int i = 0; i < n; i++) {   // x
            int u = v + j * n + i;
            R[u] = update_temperature(T, u, n, m, o, i, j, k);
        }
    }
}

int* squareDecomposition(int p, int o, int m){
    /* 2D partitioning in p parts of the rectangle of size o*m.
    Return (a, b) such that a*b = p and a<m and b<o.
    Suppose m < o.
    */
    int* result = malloc(2 * sizeof(*result));

    int a = sqrt(p);
    while(a > m || p%a != 0){
        a--;
    }
    int b = p/a;

    result[0] = b;
    result[1] = a;
    fprintf(stderr, "a = %d, b = %d \n", a, b);
    return result;
}

int* cubeDecomposition(int p, int o, int m, int n){
    /* 3D partitioning in p parts of the parallelepiped of size o*n*m.
    Return (a, b, c) such that a*b*c = p, a<o, b<n and c<m.
    Suppose m < o < n.
    */
    int* result = malloc(3 * sizeof(*result));

    int a = cbrt(p);
    while (a > m || p%a != 0){
        a--;
    }

    int b = sqrt(p/a);
    while(b > o || (p/a)%b != 0){
        b--;
    }

    int c = p/(a*b);

    result[0] = b;
    result[1] = a;
    result[2] = c;
    fprintf(stderr, "a = %d, b = %d, c = %d \n", a, b, c);
    return result;

}


void oneDimension(int my_rank, int p, int tag, MPI_Status status){

    CPU_surface = CPU_contact_surface();
    double V = L * l * E;
    int n = ceil(L / dl);
    int m = ceil(E / dl);
    int o = ceil(l / dl);
    if(my_rank == 0) {
        fprintf(stderr, "HEATSINK\n");
        fprintf(stderr, "\tDimension (cm) [x,y,z] = %.1f x %.1f x %.1f\n", 100 * L, 100 * E, 100 * l);
        fprintf(stderr, "\tVolume = %.1f cm^3\n", V * 1e6);
        fprintf(stderr, "\tWeight = %.2f kg\n", V * sink_density);
        fprintf(stderr, "\tPrice = %.2f €\n", V * sink_density * euros_per_kg);
        fprintf(stderr, "SIMULATION\n");
        fprintf(stderr, "\tGrid (x,y,z) = %d x %d x %d (%.1fMo)\n", n, m, o, 7.6293e-06 * n * m * o);
        fprintf(stderr, "\tdt = %gs\n", dt);
        fprintf(stderr, "CPU\n");
        fprintf(stderr, "\tPower = %.0fW\n", CPU_TDP);
        fprintf(stderr, "\tArea = %.1f cm^2\n", CPU_surface * 10000);
    }

    int h;  //Number of slices to calculate for the first p-1 processes
    int coupeProfondeur; // Number of slices to calculate for the last process
    if (o%p == 0) {
        h = o/p;
        coupeProfondeur = h;
    } else {
        h = o/p + 1;
        coupeProfondeur = o % h;
    }

    int profondeurTotal;
    int profondeurNext, profondeurPrev;

    double *T;
    double *R;
    if (p == 1) {
        // one process
        profondeurTotal = h;
        profondeurNext = 0;
        profondeurPrev = 0;

    } else {
        if(my_rank == 0) {

            profondeurTotal = h+1;
            profondeurNext = 1;
            profondeurPrev = 0;

        } else if (my_rank == p-1) {

            profondeurTotal = coupeProfondeur+1;
            profondeurNext = 0;
            profondeurPrev = 1;

        } else {

            profondeurTotal = h+2;
            profondeurNext = 1;
            profondeurPrev = 1;

        }
    }

    /* temperature of each cell, in degree Kelvin. */
    // INITIALISATION
    /* initially the heatsink is at the temperature of the water-cooling fluid */

    T = malloc(sizeof(*T) * profondeurTotal * n * m);
    R = malloc(sizeof(*R) * profondeurTotal * n * m);

    if (T == NULL || R == NULL) {
        perror("T or R could not be allocated");
        exit(1);
    }

    /* let's go! we switch the CPU on and launch the simulation until it reaches a stationary state. */
    double t = 0;
    int n_steps = 0;
    int convergence = 0;

    int startRead_z = 0;
    if(profondeurPrev == 1){
        startRead_z = 1;
    }
    int endRead_z = profondeurTotal;
    if(profondeurNext == 1){
        endRead_z -= 1;
    }


    // variable initialisation
    for (int u = 0; u < profondeurTotal * m * n; u++){
        R[u] = T[u] = watercooling_T + 273.15;
    }
    double debut;
    if (my_rank == 0){
        /* Start of the timer */
        debut = my_gettimeofday();
    }
    /* simulating time steps */
    while (convergence == 0) {
        /* Update all cells. xy planes are processed, for increasing values of z. */

        // Communication
        if (profondeurNext == 1) {
            MPI_Sendrecv(&T[(profondeurTotal-2)*n*m], n*m, MPI_DOUBLE,
                         my_rank+1, tag, &T[(profondeurTotal-1)*n*m], n*m,
                         MPI_DOUBLE, my_rank+1, tag, MPI_COMM_WORLD, &status);
        }
        if (profondeurPrev == 1) {
            MPI_Sendrecv(&T[1*n*m], n*m, MPI_DOUBLE, my_rank-1, tag, &T[0*n*m], n*m,
                         MPI_DOUBLE, my_rank-1, tag, MPI_COMM_WORLD, &status);
        }

        // launch simulation
        for (int k = 0; k < profondeurTotal; k++) {   // z
            int v = k * n * m;

            if (( profondeurNext == 1 && k == profondeurTotal-1) || (profondeurPrev == 1 && k == 0)) {
                // no communication case
                continue;
            }

            int K;  // depth
            if (profondeurPrev == 1) { 
                K = h*my_rank + k-1;
            } else {
                K = k;
            }

            do_xy_plane(T, R, v, n, m, o, K);
        }
        /* each second, we test the convergence, and print a short progress report */
        if (n_steps % ((int)(1 / dt)) == 0) {
            double delta_T = 0;
            double max = -INFINITY;

            for (int u = startRead_z*n*m; u < endRead_z*n*m; u++) {
                delta_T += (R[u] - T[u]) * (R[u] - T[u]);
                if (R[u] > max)
                    max = R[u];
            }

            double res = delta_T;
            double maxproc = -INFINITY;
            MPI_Reduce(&max, &maxproc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&delta_T, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if(my_rank == 0){
                delta_T = sqrt(res) / dt;
                fprintf(stderr, "t = %.1fs ; T_max = %.1f°C ; convergence = %g\n", t, maxproc - 273.15, delta_T);
                if (delta_T < 0.1)
                    convergence = 1;
            }
        }
        MPI_Bcast(&convergence, 1, MPI_INT, 0, MPI_COMM_WORLD);
        /* the new temperatures are in R */
        double *tmp = R;
        R = T;
        T = tmp;
        t += dt;
        n_steps += 1;
    }
    if(my_rank == 0) {
        fprintf(stderr, "temps = %f\n", my_gettimeofday() - debut);
    }

    double *result;
    if (my_rank == 0) {
        result = malloc(n*m*o*sizeof(*result));
    }

    int* recvcount = (int*) malloc(sizeof(int) * p); // size
    int* displs = calloc(p, sizeof(int)); // Offset
    for (int i = 0; i < p; i++){
        recvcount[i] = h*n*m;
        for (int j = 0; j < i; j++){
            displs[i] += recvcount[j];
        }
	}

    if (o%p == 0) {
        recvcount[p-1] = h*n*m;
    } else {
        recvcount[p-1] = coupeProfondeur*n*m;
    }

    if (my_rank == 0) {
        MPI_Gatherv(&T[0], recvcount[my_rank], MPI_DOUBLE, result, recvcount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(&T[m*n], recvcount[my_rank], MPI_DOUBLE, result, recvcount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (my_rank == 0) {
        #ifdef DUMP_STEADY_STATE
            printf("###### STEADY STATE; t = %.1f\n", t);
            for (int k = 0; k < o; k++) {   // z
                printf("# z = %g\n", k * dl);
                for (int j = 0; j < m; j++) {   // y
                    for (int i = 0; i < n; i++) {   // x
                        printf("%.1f ", result[k * n * m + j * n + i] - 273.15);
                    }
                    printf("\n");
                }
            }
            printf("\n");
            fprintf(stderr, "For graphical rendering: python3 rendu_picture_steady.py [filename.txt] %d %d %d\n", n, m, o);
        #endif
    }
}

// 2D implementation
void twoDimension(int y, int z, int my_rank, int p, int tag, MPI_Status status){
    /* Partitions the 3D space based on the 2 axes: Y and Z.
    int y: Number of partitions along the Y axis
    int z: Number of partitions along the Z axis
    */
    CPU_surface = CPU_contact_surface();
    double V = L * l * E;
    int n = ceil(L / dl);
    int m = ceil(E / dl);
    int o = ceil(l / dl);
    if (my_rank == 0) {
        fprintf(stderr, "HEATSINK\n");
        fprintf(stderr, "\tDimension (cm) [x,y,z] = %.1f x %.1f x %.1f\n", 100 * L, 100 * E, 100 * l);
        fprintf(stderr, "\tVolume = %.1f cm^3\n", V * 1e6);
        fprintf(stderr, "\tWeight = %.2f kg\n", V * sink_density);
        fprintf(stderr, "\tPrice = %.2f €\n", V * sink_density * euros_per_kg);
        fprintf(stderr, "SIMULATION\n");
        fprintf(stderr, "\tGrid (x,y,z) = %d x %d x %d (%.1fMo)\n", n, m, o, 7.6293e-06 * n * m * o);
        fprintf(stderr, "\tdt = %gs\n", dt);
        fprintf(stderr, "CPU\n");
        fprintf(stderr, "\tPower = %.0fW\n", CPU_TDP);
        fprintf(stderr, "\tArea = %.1f cm^2\n", CPU_surface * 10000);
    }

    int profondeur;  // Number of Z slices to calculate for the first p-1 processes
    int hauteur;  // Number of Y slices to calculate for the first p-1 processes
    int profondeur_last;  // Number of Y slices to calculate for the last process
    int hauteur_last;  // Number of Z slices to calculate for the last process


    // depth = sup (o/z).
    if (o%z == 0) {
        profondeur = o/z;
        profondeur_last = o/z;
    } else {
        profondeur = o/z + 1;
        profondeur_last = o % profondeur;
    }

    if (m%y == 0) {
        hauteur = m/y;
        hauteur_last = m/y;
    } else {
        hauteur = m/y + 1;
        hauteur_last = m % hauteur;
    }

    int profondeurTotal;  // Number of depths for each process
    int hauteurTotal;  // Number of heights for each process
    int profondeurNext;  // Number of neighbors at the next depth
    int profondeurPrev;  // Number of neighbors at the previous depth
    int hauteurNext;
    int hauteurPrev;

    // memory allocation
    double *T;
    double *R;

    for (int k = 0; k < z; k++){
        if (p == 1){
            // one process
            profondeurTotal = profondeur;
            hauteurTotal = hauteur;
            profondeurNext = 0;
            profondeurPrev = 0;
            hauteurNext = 0;
            hauteurPrev = 0;
            break;
        }
        for (int j = 0; j < y; j++){

            // Each process is associated with a part of the domain
            if (my_rank != k*y +j){
                continue;
            }


            if ((k == 0 && j == 0) || (k == z-1 && j == y-1) || (k == 0 && j == y-1) || (k == z-1 && j == 0)) {
                if (k == 0 && j == 0) {
                    profondeurTotal = profondeur + 1;
                    hauteurTotal = hauteur + 1;
                    profondeurNext = 1;
                    profondeurPrev = 0;
                    hauteurNext = 1;
                    hauteurPrev = 0;
                } else if (k == z-1 && j == y-1) {
                    profondeurTotal = profondeur_last + 1;
                    hauteurTotal = hauteur_last + 1;
                    profondeurNext = 0;
                    profondeurPrev = 1;
                    hauteurNext = 0;
                    hauteurPrev = 1;
                } else if (k == 0 && j == y-1) {
                    profondeurTotal = profondeur + 1;
                    hauteurTotal = hauteur_last + 1;
                    profondeurNext = 1;
                    profondeurPrev = 0;
                    hauteurNext = 0;
                    hauteurPrev = 1;
                } else {  // k == z-1 && j == 0
                    profondeurTotal = profondeur_last + 1;
                    hauteurTotal = hauteur + 1;
                    profondeurNext = 0;
                    profondeurPrev = 1;
                    hauteurNext = 1;
                    hauteurPrev = 0;
                }
            }

            else if ((k == 0 || k == z-1) && (j != 0) && (j != y-1)) {
                if (k == 0) {
                    profondeurTotal = profondeur + 1;
                    hauteurTotal = hauteur + 2;
                    profondeurNext = 1;
                    profondeurPrev = 0;
                    hauteurNext = 1;
                    hauteurPrev = 1;
                } else {  // k == z-1
                    profondeurTotal = profondeur_last + 1;
                    hauteurTotal = hauteur + 2;
                    profondeurNext = 0;
                    profondeurPrev = 1;
                    hauteurNext = 1;
                    hauteurPrev = 1;
                }
            }

            else if ((k != 0) && (k != z-1)  && (j == 0 || j == y-1)) {
                if (j == 0){
                    profondeurTotal = profondeur + 2;
                    hauteurTotal = hauteur + 1;
                    profondeurNext = 1;
                    profondeurPrev = 1;
                    hauteurNext = 1;
                    hauteurPrev = 0;
                }
                else {  // j == y-1
                    profondeurTotal = profondeur + 2;
                    hauteurTotal = hauteur_last + 1;
                    profondeurNext = 1;
                    profondeurPrev = 1;
                    hauteurNext = 0;
                    hauteurPrev = 1;
                }
            }

            else {
                profondeurTotal = profondeur + 2;
                hauteurTotal = hauteur + 2;
                profondeurNext = 1;
                profondeurPrev = 1;
                hauteurNext = 1;
                hauteurPrev = 1;
            }

        }
    }

    T = malloc(sizeof(*T) * profondeurTotal * n * hauteurTotal);
    R = malloc(sizeof(*R) * profondeurTotal * n * hauteurTotal);


    if (T == NULL || R == NULL) {
        perror("T or R could not be allocated");
        exit(1);
    }


    // variable initialisation
    for (int u = 0; u < profondeurTotal * hauteurTotal * n; u++){
        R[u] = T[u] = watercooling_T + 273.15;
    }



    double t = 0;
    int n_steps = 0;
    int convergence = 0;

    // Calculation of convergence

    double* buffer = malloc(sizeof(*buffer)* profondeurTotal * n);
    double* buffer2 = malloc(sizeof(*buffer2)* profondeurTotal * n);

    int startRead_z = 0;
    if(profondeurPrev == 1){
        startRead_z = 1;
    }
    int endRead_z = profondeurTotal;
    if(profondeurNext == 1){
        endRead_z -= 1;
    }
    int profondeurRead = endRead_z - startRead_z;

    int start_y, end_y;
    if (hauteurPrev == 1){
        start_y = 1;
    }
    else{
        start_y = 0;
    }

    if(hauteurNext == 1){
        end_y = hauteurTotal-1;
    }
    else{
        end_y = hauteurTotal;
    }
    int hauteurRead = end_y - start_y;
    double debut;
    if (my_rank == 0){
        /* start of the timer*/
        debut = my_gettimeofday();
    }

    while (convergence == 0) {

        if (profondeurPrev == 1){ 
        MPI_Sendrecv(&T[1 * hauteurTotal * n + start_y * n + 0], n*hauteurRead, MPI_DOUBLE,
                     my_rank-y, tag, &T[0 * hauteurTotal * n + start_y * n + 0], n*hauteurRead,
                     MPI_DOUBLE, my_rank-y, tag, MPI_COMM_WORLD, &status);
        }

        if (profondeurNext == 1){
        MPI_Sendrecv(&T[(profondeurTotal-2) * hauteurTotal * n + start_y * n + 0], n*hauteurRead, MPI_DOUBLE,
                     my_rank+y, tag, &T[(profondeurTotal-1) * hauteurTotal * n + start_y * n + 0], n*hauteurRead,
                     MPI_DOUBLE, my_rank+y, tag, MPI_COMM_WORLD, &status);
        }

        if (hauteurPrev == 1){ 

            int incr =0;
            for(int k = startRead_z; k < endRead_z;k++){
                for(int i = 0; i< n;i++){
                    buffer[incr] = T[k*hauteurTotal*n + 1*n + i];
                    incr++;
                }
            }

            MPI_Sendrecv_replace(&buffer[0], n*profondeurRead, MPI_DOUBLE,
                my_rank-1, tag, my_rank-1, tag, MPI_COMM_WORLD, &status);
            incr = 0;

            for(int k = startRead_z; k < endRead_z;k++){
                for(int i = 0; i < n;i++){
                    T[k*hauteurTotal*n + 0*n + i] = buffer[incr];
                    incr++;
                }
            }
        }

        if (hauteurNext == 1){ 

            int incr =0;
            for(int k = startRead_z; k < endRead_z;k++){
                for(int i = 0; i < n;i++){
                    buffer[incr] = T[k*hauteurTotal*n + (hauteurTotal - 2)*n + i];
                    incr++;
                }
            }

            MPI_Sendrecv_replace(&buffer[0], n*profondeurRead, MPI_DOUBLE,
                my_rank+1, tag, my_rank+1, tag, MPI_COMM_WORLD, &status);

            incr = 0;
            for(int k = startRead_z; k< endRead_z;k++){
                for(int i = 0; i< n;i++){
                    T[k*hauteurTotal*n + (hauteurTotal-1)*n + i] = buffer[incr];
                    incr++;
                }
            }
        }

        for (int k = 0; k < profondeurTotal; k++) {  // z
            int v = k * n * hauteurTotal;

            int K;
            if (profondeurPrev == 1) {

                K = (my_rank/y)*profondeur + k-1;
            } else {
                K = k;
            }
            if (K == 0 || (profondeurPrev == 1 && k == 0) || (profondeurNext == 1 && k == profondeurTotal - 1)) {

                // we do not modify the z = 0 plane: it is maintained at constant temperature via water-cooling
                continue;
            }

            for (int j = 0; j < hauteurTotal; j++) {   // y

                int J;
                if (hauteurPrev == 1) {
                    J = (my_rank%y)*hauteur + j-1;

                } else {
                    J = j;
                }

                if((hauteurPrev == 1 && j == 0) || (hauteurNext == 1 && j == hauteurTotal - 1)){
                    continue;
                }

                for (int i = 0; i < n; i++) {   // x
                    int u = v + j * n + i;

                    R[u] = update_temperature2D(T, u, n, m, o, i, J, K, hauteurTotal);
                }
            }
        }

        if (n_steps % ((int)(1 / dt)) == 0) {
            double delta_T = 0;
            double max = -INFINITY;
            double res = 0;

            for (int k = startRead_z; k < endRead_z; k++) {
                for (int j = start_y; j < end_y; j++){
                    for (int i = 0; i < n; i++) {
                        int u = (k * hauteurTotal * n) + (j * n) + i;
                        delta_T += (R[u] - T[u]) * (R[u] - T[u]);

                        if (R[u] > max)
                            max = R[u];
                    }
                }
            }

            res = delta_T;
            double maxproc = -INFINITY;
            MPI_Reduce(&max, &maxproc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&delta_T, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if(my_rank == 0){
                delta_T = sqrt(res) / dt;
                fprintf(stderr, "t = %.1fs ; T_max = %.1f°C ; convergence = %g\n", t, maxproc - 273.15, delta_T);
                if (delta_T < 0.1)
                    convergence = 1;
            }

        }
        MPI_Bcast(&convergence, 1, MPI_INT, 0, MPI_COMM_WORLD);
        double *tmp = R;
        R = T;
        T = tmp;
        t += dt;
        n_steps += 1;
    }

    if(my_rank == 0) {
        fprintf(stderr, "temps = %f\n", my_gettimeofday() - debut);
    }

    free(buffer);
    free(buffer2);

    double *result;
    if (my_rank == 0){
        result = malloc(n*m*o*sizeof(*result));
    }

    if (my_rank != 0) {
        int start_z, end_z;
        if (profondeurPrev == 1)
            start_z = 1;
        else
            start_z = 0;

        if (profondeurNext == 1)
            end_z = profondeurTotal - 1;
        else
            end_z = profondeurTotal;

        int P = end_z - start_z;

        int start_y, end_y;
        if(hauteurPrev == 1)
            start_y = 1;
        else
            start_y = 0;

        if(hauteurNext == 1)
            end_y = hauteurTotal - 1;
        else
            end_y = hauteurTotal;

        int hauteurWrite = end_y - start_y;

        MPI_Send(&P, 1, MPI_INT, 0, 999, MPI_COMM_WORLD);
        MPI_Send(&hauteurWrite, 1, MPI_INT, 0, 998, MPI_COMM_WORLD);
        MPI_Send(&start_z, 1, MPI_INT, 0, 997, MPI_COMM_WORLD);
        MPI_Send(&end_z, 1, MPI_INT, 0, 996, MPI_COMM_WORLD);

        for (int k = start_z; k < end_z; k++){
            MPI_Send(&T[k*hauteurTotal*n +start_y*n], hauteurWrite*n, MPI_DOUBLE, 0, k, MPI_COMM_WORLD);
        }

    }

    else {
        for (int k = 0; k < z; k++){
            for (int j = 0; j < y; j++){
                int rank = k*y + j;
                int P = 0; 
                int H = 0; 
                int S = 0; 
                int E = 0; 
                if (k != 0 || j != 0){
                    MPI_Recv(&P, 1, MPI_INT, rank, 999, MPI_COMM_WORLD, &status);
                    MPI_Recv(&H, 1, MPI_INT, rank, 998, MPI_COMM_WORLD, &status);
                    MPI_Recv(&S, 1, MPI_INT, rank, 997, MPI_COMM_WORLD, &status);
                    MPI_Recv(&E, 1, MPI_INT, rank, 996, MPI_COMM_WORLD, &status);

                    for (int i = S; i < E; i++){
                        int prof;
                        if(rank >=y){
                            prof = (rank/y)*(o/z) + i-1;
                        }else{
                            prof = (rank/y)*(o/z) + i;
                        }
                        int haut = (rank%y)*(m/y);
                        MPI_Recv(&result[prof*m*n + haut*n], H*n, MPI_DOUBLE, rank, i, MPI_COMM_WORLD, &status);
                    }
                } else {
                    int hauteur = hauteurTotal - hauteurNext;
                    int profondeur = profondeurTotal - profondeurNext;
                    for (int k = 0; k < profondeur; k++){
                        for (int j = 0; j < hauteur; j++){
                            for (int i = 0; i < n; i++){
                                result[k*n*m + j*n + i] = T[k*n*hauteurTotal + j*n + i];
                            }
                        }
                    }

                }

            }
        }



    }

    // write
    if (my_rank == 0){
        #ifdef DUMP_STEADY_STATE
            printf("###### STEADY STATE; t = %.1f\n", t);
            for (int k = 0; k < o; k++) {   // z
                printf("# z = %g\n", k * dl);
                for (int j = 0; j < m; j++) {   // y
                    for (int i = 0; i < n; i++) {   // y
                        printf("%.1f ", result[k * n * m + j * n + i] - 273.15);
                    }
                    printf("\n");
                }
            }
            printf("\n");
            fprintf(stderr, "For graphical rendering: python3 rendu_picture_steady.py [filename.txt] %d %d %d\n", n, m, o);
        #endif
    }
}

// 3D implementation
void threeDimension(int x, int y, int z, int my_rank, int p, int tag, MPI_Status status){
    /* Partitions the 3D space based on the Y and Z axes.
    int y: Number of partitions along the Y axis
    int z: Number of partitions along the Z axis
    int x: Number of partitions along the X axis
    */
    CPU_surface = CPU_contact_surface();
    double V = L * l * E;
    int n = ceil(L / dl);
    int m = ceil(E / dl);
    int o = ceil(l / dl);
    if (my_rank == 0) {
        fprintf(stderr, "HEATSINK\n");
        fprintf(stderr, "\tDimension (cm) [x,y,z] = %.1f x %.1f x %.1f\n", 100 * L, 100 * E, 100 * l);
        fprintf(stderr, "\tVolume = %.1f cm^3\n", V * 1e6);
        fprintf(stderr, "\tWeight = %.2f kg\n", V * sink_density);
        fprintf(stderr, "\tPrice = %.2f €\n", V * sink_density * euros_per_kg);
        fprintf(stderr, "SIMULATION\n");
        fprintf(stderr, "\tGrid (x,y,z) = %d x %d x %d (%.1fMo)\n", n, m, o, 7.6293e-06 * n * m * o);
        fprintf(stderr, "\tdt = %gs\n", dt);
        fprintf(stderr, "CPU\n");
        fprintf(stderr, "\tPower = %.0fW\n", CPU_TDP);
        fprintf(stderr, "\tArea = %.1f cm^2\n", CPU_surface * 10000);
    }
    int longueur; 
    int profondeur;  
    int hauteur; 
    int longueur_last; 
    int profondeur_last;  
    int hauteur_last; 

    if (o%z == 0) {
        profondeur = o/z;
        profondeur_last = o/z;
    } else {
        profondeur = o/z + 1;
        profondeur_last = o % profondeur;
    }


    if (m%y == 0) {
        hauteur = m/y;
        hauteur_last = m/y;
    } else {
        hauteur = m/y + 1;
        hauteur_last = m % hauteur;
    }

    if(n%x == 0){
        longueur = n/x;
        longueur_last = n/x;
    } else {
        longueur = n/x + 1;
        longueur_last = n % longueur;
    }

    int longueurTotal; 
    int profondeurTotal;  
    int hauteurTotal;
    int longueurNext;  
    int longueurPrev;   
    int profondeurNext;  
    int profondeurPrev;  
    int hauteurNext;
    int hauteurPrev;

    // memory allocation
    double *T;
    double *R;

    for (int k = 0; k < z; k++){
        if (p == 1){
            // one process
            profondeurTotal = profondeur;
            hauteurTotal = hauteur;
            longueurTotal = longueur;
            profondeurNext = 0;
            profondeurPrev = 0;
            hauteurNext = 0;
            hauteurPrev = 0;
            longueurNext = 0;
            longueurPrev = 0;
            break;
        }
        for (int j = 0; j < y; j++){
            for(int i = 0; i < x; i++){
                if (my_rank != k*y*x +j*x + i){
                    continue;
                }

                if((k == 0 && i == 0 && j == 0) || (k == 0 && i == x-1 && j == 0) || (k == z-1 && i == 0 && j == 0) || (k == z-1 && i == x-1 && j == 0) ){
                    if(k == 0 && i == 0 && j == 0){
                        profondeurTotal = profondeur + 1;
                        longueurTotal = longueur + 1;
                        hauteurTotal = hauteur + 1;
                        profondeurNext = 1;
                        profondeurPrev = 0;
                        longueurNext = 1;
                        longueurPrev = 0;
                        hauteurNext = 1;
                        hauteurPrev = 0;

                    } else if (k == 0 && i == x-1 && j == 0) {
                        profondeurTotal = profondeur + 1;
                        longueurTotal = longueur_last + 1;
                        hauteurTotal = hauteur + 1;
                        profondeurNext = 1;
                        profondeurPrev = 0;
                        hauteurNext = 1;
                        hauteurPrev = 0;
                        longueurNext = 0;
                        longueurPrev = 1;

                    } else if (k == z-1 && i == 0 && j == 0){
                        profondeurTotal = profondeur_last + 1;
                        longueurTotal = longueur + 1;
                        hauteurTotal = hauteur + 1;
                        profondeurNext = 0;
                        profondeurPrev = 1;
                        hauteurNext = 1;
                        hauteurPrev = 0;
                        longueurNext = 1;
                        longueurPrev = 0;

                    // (k == z-1 && i == x-1 && j == 0)
                    } else {
                        profondeurTotal = profondeur_last + 1;
                        longueurTotal = longueur_last + 1;
                        hauteurTotal = hauteur + 1;
                        profondeurNext = 0;
                        profondeurPrev = 1;
                        hauteurNext = 1;
                        hauteurPrev = 0;
                        longueurNext = 0;
                        longueurPrev = 1;

                    }

                } else if ((k == 0 && i == 0 && j == y-1) || (k == 0 && i == x-1 && j == y-1) || (k == z-1 && i == 0 && j == y-1) || (k == z-1 && i == x-1 && j == y-1) ) {
                    if (k == 0 && i == 0 && j == y-1) {
                        profondeurTotal = profondeur + 1;
                        longueurTotal = longueur + 1;
                        hauteurTotal = hauteur_last + 1;
                        profondeurNext = 1;
                        profondeurPrev = 0;
                        longueurNext = 1;
                        longueurPrev = 0;
                        hauteurNext = 0;
                        hauteurPrev = 1;

                    } else if (k == 0 && i == x-1 && j == y-1) {
                        profondeurTotal = profondeur + 1;
                        longueurTotal = longueur_last + 1;
                        hauteurTotal = hauteur_last + 1;
                        profondeurNext = 1;
                        profondeurPrev = 0;
                        longueurNext = 0;
                        longueurPrev = 1;
                        hauteurNext = 0;
                        hauteurPrev = 1;

                    } else if (k == z-1 && i == 0 && j == y-1) {
                        profondeurTotal = profondeur_last + 1;
                        longueurTotal = longueur + 1;
                        hauteurTotal = hauteur_last + 1;
                        profondeurNext = 0;
                        profondeurPrev = 1;
                        hauteurNext = 0;
                        hauteurPrev = 1;
                        longueurNext = 1;
                        longueurPrev = 0;

                    // (k == z-1 && i == x-1 && j == y-1)
                    } else {
                        profondeurTotal = profondeur_last + 1;
                        longueurTotal = longueur_last + 1;
                        hauteurTotal = hauteur_last + 1;
                        profondeurNext = 0;
                        profondeurPrev = 1;
                        hauteurNext = 0;
                        hauteurPrev = 1;
                        longueurNext = 0;
                        longueurPrev = 1;

                    }

                } else if ((k == 0 || k == z-1) && (j != 0) && (j != y-1) && (i == 0 || i == x-1)) {
                    if( k == 0 ){
                        if ( i == 0 ){
                            profondeurTotal = profondeur + 1;
                            longueurTotal = longueur + 1;
                            hauteurTotal = hauteur + 2;
                            profondeurNext = 1;
                            profondeurPrev = 0;
                            hauteurNext = 1;
                            hauteurPrev = 1;
                            longueurNext = 1;
                            longueurPrev = 0;

                        } else if ( i == x-1 ) {
                            profondeurTotal = profondeur + 1;
                            longueurTotal = longueur_last + 1;
                            hauteurTotal = hauteur + 2;
                            profondeurNext = 1;
                            profondeurPrev = 0;
                            hauteurNext = 1;
                            hauteurPrev = 1;
                            longueurNext = 0;
                            longueurPrev = 1;
                        }
                    } if ( k == z-1 ) {
                        if ( i == 0 ){
                            profondeurTotal = profondeur_last + 1;
                            longueurTotal = longueur + 1;
                            hauteurTotal = hauteur + 2;
                            profondeurNext = 0;
                            profondeurPrev = 1;
                            hauteurNext = 1;
                            hauteurPrev = 1;
                            longueurNext = 1;
                            longueurPrev = 0;

                        } else if ( i == x-1 ) {
                            profondeurTotal = profondeur_last + 1;
                            longueurTotal = longueur_last + 1;
                            hauteurTotal = hauteur + 2;
                            profondeurNext = 0;
                            profondeurPrev = 1;
                            hauteurNext = 1;
                            hauteurPrev = 1;
                            longueurNext = 0;
                            longueurPrev = 1;
                        }
                    }
                } else if ((k == 0 || k == z-1) && (i != 0 || i != x-1)) {
                    if (k == 0) {
                        if (j == 0) {
                            profondeurTotal = profondeur + 1;
                            longueurTotal = longueur + 2;
                            hauteurTotal = hauteur + 1;
                            profondeurNext = 1;
                            profondeurPrev = 0;
                            hauteurNext = 1;
                            hauteurPrev = 0;
                            longueurNext = 1;
                            longueurPrev = 1;

                        } else if (j == y-1) {
                            profondeurTotal = profondeur + 1;
                            longueurTotal = longueur + 2;
                            hauteurTotal = hauteur_last + 1;
                            profondeurNext = 1;
                            profondeurPrev = 0;
                            hauteurNext = 0;
                            hauteurPrev = 1;
                            longueurNext = 1;
                            longueurPrev = 1;

                        } else {
                            profondeurTotal = profondeur + 1;
                            longueurTotal = longueur + 2;
                            hauteurTotal = hauteur + 2;
                            profondeurNext = 1;
                            profondeurPrev = 0;
                            hauteurNext = 1;
                            hauteurPrev = 1;
                            longueurNext = 1;
                            longueurPrev = 1;
                        }
                    } else if (k == z-1) {
                        if (j == 0) {
                            profondeurTotal = profondeur_last + 1;
                            longueurTotal = longueur + 2;
                            hauteurTotal = hauteur + 1;
                            profondeurNext = 0;
                            profondeurPrev = 1;
                            hauteurNext = 1;
                            hauteurPrev = 0;
                            longueurNext = 1;
                            longueurPrev = 1;
                        } else if (j == y-1) {
                            profondeurTotal = profondeur_last + 1;
                            longueurTotal = longueur + 2;
                            hauteurTotal = hauteur_last + 1;
                            profondeurNext = 0;
                            profondeurPrev = 1;
                            hauteurNext = 0;
                            hauteurPrev = 1;
                            longueurNext = 1;
                            longueurPrev = 1;

                        } else {
                            profondeurTotal = profondeur_last + 1;
                            longueurTotal = longueur + 2;
                            hauteurTotal = hauteur + 2;
                            profondeurNext = 0;
                            profondeurPrev = 1;
                            hauteurNext = 1;
                            hauteurPrev = 1;
                            longueurNext = 1;
                            longueurPrev = 1;

                        }
                    }
                } else if ((i == 0 || i == x-1) && (k !=0 || k != z-1)) {
                    if(i == 0) {
                        if(j == 0) {
                            profondeurTotal = profondeur + 2;
                            longueurTotal = longueur + 1;
                            hauteurTotal = hauteur + 1;
                            profondeurNext = 1;
                            profondeurPrev = 1;
                            hauteurNext = 1;
                            hauteurPrev = 0;
                            longueurNext = 1;
                            longueurPrev = 0;

                        } else if(j == y-1) {
                            profondeurTotal = profondeur + 2;
                            longueurTotal = longueur + 1;
                            hauteurTotal = hauteur_last + 1;
                            profondeurNext = 1;
                            profondeurPrev = 1;
                            hauteurNext = 0;
                            hauteurPrev = 1;
                            longueurNext = 1;
                            longueurPrev = 0;

                        } else {
                            profondeurTotal = profondeur + 2;
                            longueurTotal = longueur + 1;
                            hauteurTotal = hauteur + 2;
                            profondeurNext = 1;
                            profondeurPrev = 1;
                            hauteurNext = 1;
                            hauteurPrev = 1;
                            longueurNext = 1;
                            longueurPrev = 0;

                        }
                    } else if(i == x-1) {
                        if(j == 0) {
                            profondeurTotal = profondeur + 2;
                            longueurTotal = longueur_last + 1;
                            hauteurTotal = hauteur + 1;
                            profondeurNext = 1;
                            profondeurPrev = 1;
                            hauteurNext = 1;
                            hauteurPrev = 0;
                            longueurNext = 0;
                            longueurPrev = 1;
                        } else if (j == y-1) {
                            profondeurTotal = profondeur + 2;
                            longueurTotal = longueur_last + 1;
                            hauteurTotal = hauteur_last + 1;
                            profondeurNext = 1;
                            profondeurPrev = 1;
                            hauteurNext = 0;
                            hauteurPrev = 1;
                            longueurNext = 0;
                            longueurPrev = 1;

                        } else {
                            profondeurTotal = profondeur + 2;
                            longueurTotal = longueur_last + 1;
                            hauteurTotal = hauteur + 2;
                            profondeurNext = 1;
                            profondeurPrev = 1;
                            hauteurNext = 1;
                            hauteurPrev = 1;
                            longueurNext = 0;
                            longueurPrev = 1;
                        }
                    }
                } else if ((k!= 0 || k!= z-1) && (i!= 0 || i!= x-1) && (j==0 || j == y-1)) {
                    if(j == 0) {
                        profondeurTotal = profondeur + 2;
                        longueurTotal = longueur + 2;
                        hauteurTotal = hauteur + 1;
                        profondeurNext = 1;
                        profondeurPrev = 1;
                        hauteurNext = 1;
                        hauteurPrev = 0;
                        longueurNext = 1;
                        longueurPrev = 1;
                    } else if (j == y-1) {
                        profondeurTotal = profondeur + 2;
                        longueurTotal = longueur + 2;
                        hauteurTotal = hauteur_last + 1;
                        profondeurNext = 1;
                        profondeurPrev = 1;
                        hauteurNext = 0;
                        hauteurPrev = 1;
                        longueurNext = 1;
                        longueurPrev = 1;
                    }
                }
                else {
                    profondeurTotal = profondeur + 2;
                    longueurTotal = longueur + 2;
                    hauteurTotal = hauteur + 2;
                    profondeurNext = 1;
                    profondeurPrev = 1;
                    hauteurNext = 1;
                    hauteurPrev = 1;
                    longueurNext = 1;
                    longueurPrev = 1;
                }
            }
        }
    }

    T = malloc(sizeof(*T) * profondeurTotal * longueurTotal * hauteurTotal);
    R = malloc(sizeof(*R) * profondeurTotal * longueurTotal * hauteurTotal);


    if (T == NULL || R == NULL) {
        perror("T or R could not be allocated");
        exit(1);
    }

    for (int u = 0; u < profondeurTotal * hauteurTotal * longueurTotal; u++){
        R[u] = T[u] = watercooling_T + 273.15;
    }



    double t = 0;
    int n_steps = 0;
    int convergence = 0;


    int startRead_z = 0;
    if(profondeurPrev == 1){
        startRead_z = 1;
    }
    int endRead_z = profondeurTotal;
    if(profondeurNext == 1){
        endRead_z -= 1;
    }
    int profondeurRead = endRead_z - startRead_z;

    int start_y, end_y;
    if (hauteurPrev == 1){
        start_y = 1;
    }
    else{
        start_y = 0;
    }

    if(hauteurNext == 1){
        end_y = hauteurTotal-1;
    }
    else{
        end_y = hauteurTotal;
    }
    int hauteurRead = end_y - start_y;

    int start_x = 0;
    if(longueurPrev == 1){
        start_x = 1;
    }
    int end_x = longueurTotal;
    if(longueurNext == 1){
        end_x -= 1;
    }
    int longueurRead = end_x - start_x;


    double* bufferHauteur = malloc(sizeof(*bufferHauteur)* profondeurTotal * longueurTotal);
    double* bufferLongueur = malloc(sizeof(*bufferLongueur)* profondeurTotal * hauteurTotal);
    double debut;
    if(my_rank == 0){
        /* Start of the timer */
        debut = my_gettimeofday();
    }
    while (convergence == 0) {

        if (profondeurPrev == 1){ 
            MPI_Sendrecv(&T[1 * hauteurTotal * longueurTotal + start_y * longueurTotal + 0], longueurTotal*hauteurRead, MPI_DOUBLE,
                     my_rank-(y*x), tag, &T[0 * hauteurTotal * longueurTotal + start_y * longueurTotal + 0], longueurTotal*hauteurRead,
                     MPI_DOUBLE, my_rank-(y*x), tag, MPI_COMM_WORLD, &status);
        }

        if (profondeurNext == 1){ 
            MPI_Sendrecv(&T[(profondeurTotal-2) * hauteurTotal * longueurTotal + start_y * longueurTotal + 0], longueurTotal*hauteurRead, MPI_DOUBLE,
                     my_rank+(y*x), tag, &T[(profondeurTotal-1) * hauteurTotal * longueurTotal + start_y * longueurTotal + 0], longueurTotal*hauteurRead,
                     MPI_DOUBLE, my_rank+(y*x), tag, MPI_COMM_WORLD, &status);
        }

        if (hauteurPrev == 1){

            int incr =0;
            for(int k = startRead_z; k < endRead_z;k++){
                for(int i = start_x; i< end_x;i++){
                    bufferHauteur[incr] = T[k*hauteurTotal*longueurTotal + 1*longueurTotal + i];
                    incr++;
                }
            }

            MPI_Sendrecv_replace(&bufferHauteur[0], longueurRead*profondeurRead, MPI_DOUBLE,
                my_rank-x, tag, my_rank-x, tag, MPI_COMM_WORLD, &status);
            incr = 0;

            for(int k = startRead_z; k < endRead_z;k++){
                for(int i = start_x; i < end_x;i++){
                    T[k*hauteurTotal*longueurTotal + 0*longueurTotal + i] = bufferHauteur[incr];
                    incr++;
                }
            }
        }

        if (hauteurNext == 1){

            int incr =0;
            for(int k = startRead_z; k < endRead_z;k++){
                for(int i = start_x; i < end_x;i++){
                    bufferHauteur[incr] = T[k*hauteurTotal*longueurTotal + (hauteurTotal - 2)*longueurTotal + i];
                    incr++;
                }
            }

            MPI_Sendrecv_replace(&bufferHauteur[0], longueurRead*profondeurRead, MPI_DOUBLE,
                my_rank+x, tag, my_rank+x, tag, MPI_COMM_WORLD, &status);

            incr = 0;
            for(int k = startRead_z; k< endRead_z;k++){
                for(int i = start_x; i< end_x;i++){
                    T[k*hauteurTotal*longueurTotal + (hauteurTotal-1)*longueurTotal + i] = bufferHauteur[incr];
                    incr++;
                }
            }
        }

        if(longueurPrev == 1){ 
            int incr =0;
            for(int k = startRead_z; k < endRead_z;k++){
                for(int j = start_y; j < end_y;j++){
                    bufferLongueur[incr] = T[k*hauteurTotal*longueurTotal + j*longueurTotal + 1];
                    incr++;
                }
            }

            MPI_Sendrecv_replace(&bufferLongueur[0], hauteurRead*profondeurRead, MPI_DOUBLE,
                my_rank-1, tag, my_rank-1, tag, MPI_COMM_WORLD, &status);

            incr = 0;
            for(int k = startRead_z; k< endRead_z;k++){
                for(int j = start_y; j< end_y;j++){
                    T[k*hauteurTotal*longueurTotal + j*longueurTotal + 0] = bufferLongueur[incr];
                    incr++;
                }
            }
        }

        if(longueurNext == 1){
            int incr =0;
            for(int k = startRead_z; k < endRead_z;k++){
                for(int j = start_y; j < end_y;j++){
                    bufferLongueur[incr] = T[k*hauteurTotal*longueurTotal + j*longueurTotal + longueurTotal-2];
                    incr++;
                }
            }

            MPI_Sendrecv_replace(&bufferLongueur[0], hauteurRead*profondeurRead, MPI_DOUBLE,
                my_rank+1, tag, my_rank+1, tag, MPI_COMM_WORLD, &status);

            incr = 0;
            for(int k = startRead_z; k< endRead_z;k++){
                for(int j = start_y; j< end_y;j++){
                    T[k*hauteurTotal*longueurTotal + j*longueurTotal + longueurTotal-1] = bufferLongueur[incr];
                    incr++;
                }
            }
        }

        for (int k = 0; k < profondeurTotal; k++) {  // z


            int v = k * longueurTotal * hauteurTotal;

            int K; 
            if (profondeurPrev == 1) {
                K = (my_rank/(y*x))*profondeur + k-1;
            } else {
                K = k;
            }

            if (K == 0 || (profondeurPrev == 1 && k == 0) || (profondeurNext == 1 && k == profondeurTotal - 1)) {

                // we do not modify the z = 0 plane: it is maintained at constant temperature via water-cooling
                continue;
            }
            for (int j = 0; j < hauteurTotal; j++) {   // y

                int J;
                if (hauteurPrev == 1) {
                    J = ((my_rank%(x*y))/x)*hauteur + j-1;

                } else {
                    J = j;
                }

                if((hauteurPrev == 1 && j == 0) || (hauteurNext == 1 && j == hauteurTotal - 1)){
                    continue;
                }

                for (int i = 0; i < longueurTotal; i++) {   // x
                    int u = v + j * longueurTotal + i;

                    if((longueurPrev == 1 && i == 0) || (longueurNext == 1 && i == longueurTotal - 1)){
                        continue;
                    }

                    int I;
                    if(longueurPrev == 1){
                        I = (my_rank%x) * longueur + i-1;
                    } else {
                        I = i;
                    }


                    R[u] = update_temperature3D(T, u, n, m, o, I, J, K, hauteurTotal, longueurTotal);
                }
            }
        }


         if (n_steps % ((int)(1 / dt)) == 0) {
            double delta_T = 0;
            double max = -INFINITY;
            double res = 0;

            for (int k = startRead_z; k < endRead_z; k++) {
                for (int j = start_y; j < end_y; j++){
                    for (int i = start_x; i < end_x; i++) {
                        int u = (k * hauteurTotal * longueurTotal) + (j * longueurTotal) + i;
                        delta_T += (R[u] - T[u]) * (R[u] - T[u]);

                        if (R[u] > max)
                            max = R[u];
                    }
                }
            }

            res = delta_T;
            double maxproc = -INFINITY;
            MPI_Reduce(&max, &maxproc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&delta_T, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if(my_rank == 0){
                delta_T = sqrt(res) / dt;
                fprintf(stderr, "t = %.1fs ; T_max = %.1f°C ; convergence = %g\n", t, maxproc - 273.15, delta_T);
                if (delta_T < 0.1)
                    convergence = 1;
            }

        }
        MPI_Bcast(&convergence, 1, MPI_INT, 0, MPI_COMM_WORLD);
        double *tmp = R;
        R = T;
        T = tmp;
        t += dt;
        n_steps += 1;

    }

    if (my_rank == 0) {
        fprintf(stderr, "temps = %f\n", my_gettimeofday() - debut);
        printf("temps = %f\n", my_gettimeofday() - debut);
    }

    free(bufferHauteur);
    free(bufferLongueur);


    double *result;
    if (my_rank == 0){
        result = malloc(n*m*o*sizeof(*result));
    }
    int *tab = malloc(sizeof(*tab)*7);
    int *infoTab = malloc(sizeof(*infoTab)*3);

    if (my_rank != 0) {
        tab[0] = profondeurRead;
        tab[1] = hauteurRead;
        tab[2] = longueurRead;
        tab[3] = startRead_z;
        tab[4] = endRead_z;
        tab[5] = start_x;
        tab[6] = end_x;

        infoTab[0] = longueurTotal;
        infoTab[1] = hauteurTotal;
        infoTab[2] = profondeurTotal;

        MPI_Send(&infoTab[0], 3, MPI_INT, 0, my_rank, MPI_COMM_WORLD);
        MPI_Send(&tab[0], 7, MPI_INT, 0, my_rank, MPI_COMM_WORLD);
        for (int k= startRead_z; k < endRead_z; k++) {
            MPI_Send(&T[k*hauteurTotal*longueurTotal + start_y * longueurTotal], hauteurRead*longueurTotal, MPI_DOUBLE, 0, k, MPI_COMM_WORLD);
        }

    } else {

        for(int k = 0; k < z; k++) {
            for(int j = 0; j < y; j++){
                for(int i = 0; i < x; i++){

                    int rank = k*y*x + j*x + i;
                    if(rank != 0){
                        MPI_Recv(&infoTab[0], 3, MPI_INT, rank, rank, MPI_COMM_WORLD, &status);
                        MPI_Recv(&tab[0], 7, MPI_INT, rank, rank, MPI_COMM_WORLD, &status);
                        double *tmp = malloc(sizeof(*tmp)*tab[1]*infoTab[0]);
                        for( int l = tab[3]; l < tab[4]; l++) {
                            int prof;
                            if (rank >= y*x) {
                                prof = (rank/(y*x))*profondeur + l-1;
                            } else {
                                prof = (rank/(y*x))*profondeur + l;
                            }
                            int haut = ((rank%(x*y))/x)*hauteur;
                            int line;
                            if (rank%x == 0) {
                                line = (rank%x) * longueur + i-1;
                            }else {
                                line = 0;
                            }
                            MPI_Recv(&tmp[0], tab[0]*infoTab[0], MPI_DOUBLE, rank, l, MPI_COMM_WORLD, &status);
                            for(int col = 0; col < tab[1]; col ++){
                                for(int ligne = tab[5]; ligne < tab[6]; ligne ++)
                                {
                                    result[prof*m*n + haut*n + (line + ligne)] = tmp[col*infoTab[0] + ligne];
                                }
                            }

                        }
                        free(tmp);
                    }
                }
            }
        }
    }

    if (my_rank == 0) {
        for(int k = startRead_z; k < endRead_z; k++) {
            for(int j = start_y; j < end_y; j++) {
                for(int i = start_x; i < end_x; i++) {
                    result[k*m*n + j*n + i] = T[k*hauteurTotal*longueurTotal + j*longueurTotal + i];
                }
            }
        }
    }

    // write
    if (my_rank == 0){
        #ifdef DUMP_STEADY_STATE
            printf("###### STEADY STATE; t = %.1f\n", t);
            for (int k = 0; k < o; k++) {   // z
                printf("# z = %g\n", k * dl);
                for (int j = 0; j < m; j++) {   // y
                    for (int i = 0; i < n; i++) {   // y
                        printf("%.1f ", result[k * n * m + j * n + i] - 273.15);
                    }
                    printf("\n");
                }
            }
            printf("\n");
            fprintf(stderr, "For graphical rendering: python3 rendu_picture_steady.py [filename.txt] %d %d %d\n", n, m, o);
        #endif
    }

}

int main(int argc, char* argv[]){
    int n = ceil(L / dl);
    int m = ceil(E / dl);
    int o = ceil(l / dl);

    int my_rank; /* rank of the process */
    int p;       /* number of processes */
    int tag = 0;

    MPI_Status status;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);


    /*
    partition_choice:
    3: 3D
    2: 2D
    1: 1D
    */

    int partition_choice = 3;


    if (argc >= 2) {
        if (1 <= atoi(argv[1]) && atoi(argv[1]) <= 3){
            partition_choice = atoi(argv[1]);
        }

    }

    if (my_rank == 0){
        fprintf(stderr, "PARITIONING: %dD-Partitioning \n", partition_choice);
    }



    // FIRST DIMENSION
    if (partition_choice == 1){
        oneDimension(my_rank, p, tag, status);
    }

    // SECOND DIMENSION
    if (partition_choice == 2){
        int* decompositionSquare = malloc(2 * sizeof(*decompositionSquare));
        if (my_rank == 0) {
            decompositionSquare = squareDecomposition(p, o, m);
        }
        MPI_Bcast(&decompositionSquare[0], 2, MPI_INT, 0, MPI_COMM_WORLD);
        if(decompositionSquare[1] > decompositionSquare[0]){
            twoDimension(decompositionSquare[0], decompositionSquare[1],  my_rank, p, tag, status);
        } else {
            twoDimension(decompositionSquare[1], decompositionSquare[0],  my_rank, p, tag, status);
        }
    }

    // THIRD DIMENSION
    if (partition_choice == 3){
        int* decompositionCube = malloc(3*sizeof(*decompositionCube));
        if(my_rank == 0){
                decompositionCube = cubeDecomposition(p, o, m, n);
            }
            MPI_Bcast(&decompositionCube[0], 3, MPI_INT, 0, MPI_COMM_WORLD);
            threeDimension(decompositionCube[0], decompositionCube[1], decompositionCube[2],  my_rank, p, tag, status);
    }

    MPI_Finalize();
    exit(EXIT_SUCCESS);
}
