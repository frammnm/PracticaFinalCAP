#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

//Reference implementation of Jacobi for mp2 (sequential)
//Constants are being used instead of arguments
#define BC_HOT 1.0
#define BC_COLD 0.0
#define INITIAL_GRID 0.5
#define N_DIM 9
#define MAX_ITERATIONS 1000
#define TOL 1.0e-4

struct timeval tv;
double get_clock()
{
	struct timeval tv;
	int ok;
	ok = gettimeofday(&tv, (void *)0);
	if (ok < 0)
	{
		printf("gettimeofday error");
	}
	return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

void print_array_double(int rank, double *arr, int size, char *name)
{
	int i;
	for (i = 0; i < size; i++)
	{
		printf("[%i]%s [%d]: %6.4lf \n", rank, name, i, arr[i]);
	}
}

double **create_matrix(int n)
{
	int i;
	double **a;

	a = (double **)malloc(sizeof(double *) * n);
	for (i = 0; i < n; i++)
	{
		a[i] = (double *)malloc(sizeof(double) * n);
	}

	return a;
}

void init_matrix(double **a, int n)
{

	int i, j;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
			a[i][j] = INITIAL_GRID;
	}
}

void swap_matrix(double ***a, double ***b)
{

	double **temp;

	temp = *a;
	*a = *b;
	*b = temp;
}

void print_grid(double **a, int nstart, int nend)
{

	int i, j;

	for (i = nstart; i < nend; i++)
	{
		for (j = nstart; j < nend; j++)
		{
			printf("%6.4lf ", a[i][j]);
		}
		printf("\n");
	}
}

void free_matrix(double **a, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		free(a[i]);
	}
	free(a);
}

void init_matrix_borders(int column, int row, int p, double *north, double *south, double *east, double *west)
{
	int i;

	//If im the first row
	if (row == 0)
	{
		if (column == 0)
		{
			for (i = 0; i < p; i++)
			{
				north[i] = BC_HOT;
				south[i] = INITIAL_GRID;
				east[i] = INITIAL_GRID;
				west[i] = BC_HOT;
			}
		}
		else if (column == p - 1)
		{
			for (i = 0; i < p; i++)
			{
				north[i] = BC_HOT;
				south[i] = INITIAL_GRID;
				east[i] = BC_HOT;
				west[i] = INITIAL_GRID;
			}
		}
		else
		{
			for (i = 0; i < p; i++)
			{
				north[i] = BC_HOT;
				south[i] = INITIAL_GRID;
				east[i] = INITIAL_GRID;
				west[i] = INITIAL_GRID;
			}
		}
	}
	//Check in last row
	else if (row == p - 1)
	{
		if (column == 0)
		{
			for (i = 0; i < p; i++)
			{
				north[i] = INITIAL_GRID;
				south[i] = BC_COLD;
				east[i] = INITIAL_GRID;
				west[i] = BC_HOT;
			}
		}
		else if (column == p - 1)
		{
			for (i = 0; i < p; i++)
			{
				north[i] = INITIAL_GRID;
				south[i] = BC_COLD;
				east[i] = BC_HOT;
				west[i] = INITIAL_GRID;
			}
		}
		else
		{
			for (i = 0; i < p; i++)
			{
				north[i] = INITIAL_GRID;
				south[i] = BC_COLD;
				east[i] = INITIAL_GRID;
				west[i] = INITIAL_GRID;
			}
		}
	}
	//Im not in the first or last rows
	else if (column == 0)
	{
		for (i = 0; i < p; i++)
		{
			north[i] = INITIAL_GRID;
			south[i] = INITIAL_GRID;
			east[i] = INITIAL_GRID;
			west[i] = BC_HOT;
		}
	}
	else if (column == p - 1)
	{
		for (i = 0; i < p; i++)
		{
			north[i] = INITIAL_GRID;
			south[i] = INITIAL_GRID;
			east[i] = BC_HOT;
			west[i] = INITIAL_GRID;
		}
	}
	else
	{
		for (i = 0; i < p; i++)
		{
			north[i] = INITIAL_GRID;
			south[i] = INITIAL_GRID;
			east[i] = INITIAL_GRID;
			west[i] = INITIAL_GRID;
		}
	}
}

double get_max_diff(double **a, double **b, int p, double *north, double *south, double *east, double *west)
{
	int i, j;
	double maxdiff = 0.0;

	// Compute inner submatrix new grid values
	for (i = 1; i < p - 1; i++)
	{
		for (j = 1; j < p - 1; j++)
		{
			//self + north + south + west + east
			b[i][j] = 0.2 * (a[i][j] + a[i - 1][j] + a[i + 1][j] + a[i][j - 1] + a[i][j + 1]);
			if (fabs(b[i][j] - a[i][j]) > maxdiff)
			{
				maxdiff = fabs(b[i][j] - a[i][j]);
			}
		}
	}

	// Compute border values of the submatrix

	//Row 0

	//Element m[0, 0] left top corner
	b[0][0] = 0.2 * (a[0][0] + north[0] + a[1][0] + west[0] + a[0][1]);
	if (fabs(b[0][0] - a[0][0]) > maxdiff)
	{
		maxdiff = fabs(b[0][0] - a[0][0]);
	}

	//Elements m[1, p-2]
	i = 0;
	for (j = 1; j < p - 1; j++)
	{
		b[i][j] = 0.2 * (a[i][j] + north[j] + a[i + 1][j] + a[i][j - 1] + a[i][j + 1]);
		if (fabs(b[i][j] - a[i][j]) > maxdiff)
		{
			maxdiff = fabs(b[i][j] - a[i][j]);
		}
	}

	//Element m[0, p-1] right top corner
	b[0][p - 1] = 0.2 * (a[0][p - 1] + north[p - 1] + a[1][p - 1] + a[0][j - 1] + east[0]);
	if (fabs(b[0][p - 1] - a[0][p - 1]) > maxdiff)
	{
		maxdiff = fabs(b[0][p - 1] - a[0][p - 1]);
	}

	//Column 0 m[1, p-2]
	j = 0;
	for (i = 1; i < p - 1; i++)
	{
		b[i][j] = 0.2 * (a[i][j] + a[i - 1][j] + a[i + 1][j] + west[i] + a[i][j + 1]);
		if (fabs(b[i][j] - a[i][j]) > maxdiff)
		{
			maxdiff = fabs(b[i][j] - a[i][j]);
		}
	}

	//Element m[p-1, 0] left bot corner
	b[p - 1][0] = 0.2 * (a[p - 1][0] + a[p - 2][0] + south[0] + west[p - 1] + a[p - 1][1]);
	if (fabs(b[p - 1][0] - a[p - 1][0]) > maxdiff)
	{
		maxdiff = fabs(b[p - 1][0] - a[p - 1][0]);
	}

	//Row p-1
	i = p - 1;
	for (j = 1; j < p - 1; j++)
	{
		b[i][j] = 0.2 * (a[i][j] + a[i - 1][j] + south[j] + a[i - 1][j] + a[i][j + 1]);
		if (fabs(b[i][j] - a[i][j]) > maxdiff)
		{
			maxdiff = fabs(b[i][j] - a[i][j]);
		}
	}

	//Element m[p-1, p-1] right bot corner
	b[p - 1][p - 1] = 0.2 * (a[p - 1][p - 1] + a[p - 2][p - 1] + south[p - 1] + a[p - 1][p - 2] + east[p - 2]);
	if (fabs(b[p - 1][p - 1] - a[p - 1][p - 1]) > maxdiff)
	{
		maxdiff = fabs(b[p - 1][p - 1] - a[p - 1][p - 1]);
	}

	//Column p-1 m[1, p-2]
	j = p - 1;
	for (i = 1; i < p - 1; i++)
	{
		b[i][j] = 0.2 * (a[i][j] + a[i - 1][j] + a[i + 1][j] + a[i][j - 1] + east[i]);
		if (fabs(b[i][j] - a[i][j]) > maxdiff)
		{
			maxdiff = fabs(b[i][j] - a[i][j]);
		}
	}

	return maxdiff;
}

void com_borders(int rank, int col, int row, int p, double **a, double *north, double *south, double *east, double *west, double *send_buffer, double *recv_buffer)
{
	int j;

	//If im not in first row, i have to send my top border
	if (row != 0)
	{
		for (j = 0; j < p; j++)
		{
			send_buffer[j] = a[0][j];
		}
		MPI_Send(send_buffer, p, MPI_DOUBLE, rank - p, 0, MPI_COMM_WORLD);
	}
	// If im not in the last row, ill receive a bottom border
	if (row != p - 1)
	{
		MPI_Recv(south, p, MPI_DOUBLE, rank + p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("Received bot border [%i] from [%i]:\n", rank, rank + p);
		print_array_double(rank, south, p, "southR");
	}

	//If im not in last row, i have to send my bottom border
	if (row != p - 1)
	{
		for (j = 0; j < p; j++)
		{
			send_buffer[j] = a[p][j];
		}
		MPI_Send(send_buffer, p, MPI_DOUBLE, rank + p, 0, MPI_COMM_WORLD);
	}
	//If im in not ii the first row, ill receive a top border
	if (row != 0)
	{
		MPI_Recv(north, p, MPI_DOUBLE, rank - p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("Received top border [%i] from [%i]:\n", rank, rank - p);
		print_array_double(rank, north, p, "northR");
	}

	//If im not in first column, i have to send my left border
	if (col != 0)
	{
		for (j = 0; j < p; j++)
		{
			send_buffer[j] = a[j][0];
		}

		MPI_Send(send_buffer, p, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
	}
	//If im not in the last column, ill receive a right border
	if (col != p - 1)
	{
		MPI_Recv(east, p, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	//If im not in last column, I have to send my right border
	if (col != p - 1)
	{
		for (j = 0; j < p; j++)
		{
			send_buffer[j] = a[j][p];
		}

		MPI_Send(send_buffer, p, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
	}
	//If im not in the first column, ill receive a left border
	if (col != 0)
	{
		MPI_Recv(west, p, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}

void fill_res(double **a, int rank, int n, double **res)
{
	int i, j;
	int p = (int)sqrt(n);
	int col = rank % ((int)sqrt(n)) * p;
	int row = (int)(rank / (int)sqrt(n)) * p;

	for (i = 0; i < p; i++)
	{
		for (j = 0; j < p; j++)
		{
			res[row + i][col + j] = a[i][j];
		}
	}
}

void get_results(int rank, int n, double **a, double **b, double **res)
{
	// Receive results from other nodes
	if (rank == 0)
	{
		// MPI_Send(a[0], n * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		// MPI_Recv(res[0], 1, double_strided_vect, root_rank, generic_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		for (int i = 1; i < n; i++)
		{
			MPI_Recv(b, 9, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			fill_res(b, i, n, res);
		}
	}
	else
	{
		// Send my results to root
		MPI_Send(a, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
}

int main(int argc, char *argv[])
{
	int i, j, p, n, iteration, my_column, my_row;
	double **a, **b, **result, maxdiff;
	double tstart, tend, ttotal;

	if (argc < 1)
	{
		printf("Missing argument\n");
		exit(-1);
	}

	// Problem params
	n = atoi(argv[1]); //Number of proccesses
	p = (int)sqrt(n);  //Size of small matrix

	if ((p * p) != n)
	{
		printf("Argument is not a perfect square\n");
		exit(-1);
	}

	// Initialize the MPI environment
	// MPI_Init(NULL, NULL);
	MPI_Init(&argc, &argv);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Where am i in the bigger matrix?
	my_row = (int)(world_rank / (int)sqrt(n));
	my_column = world_rank % ((int)sqrt(n));

	//Matrix for operations on each step
	a = create_matrix(p);
	b = create_matrix(p);
	if (world_rank == 0)
	{
		result = create_matrix(n);
		init_matrix(result, n);
	}

	//Sides for the submatrices of each proccess
	double *north, *south, *east, *west;
	north = malloc(sizeof(double) * p);
	south = malloc(sizeof(double) * p);
	east = malloc(sizeof(double) * p);
	west = malloc(sizeof(double) * p);

	double *send_buffer = malloc(sizeof(double) * p);
	double *recv_buffer = malloc(sizeof(double) * p);

	//Sides for each submatrix (could be hot, cold or usual data)
	init_matrix_borders(my_column, my_row, p, north, south, east, west);
	init_matrix(a, p);

	// Copy a to b
	for (i = 0; i < p; i++)
	{
		for (j = 0; j < p; j++)
		{
			b[i][j] = a[i][j];
		}
	}

	// Main simulation routine
	iteration = 0;
	maxdiff = 1.0;

	if (world_rank == 0)
	{
		printf("Running simulation with tolerance=%lf and max iterations=%d\n",
			   TOL, MAX_ITERATIONS);
	}

	tstart = MPI_Wtime();
	// tstart = get_clock();
	int iter = 0;
	while (maxdiff > TOL && iteration < MAX_ITERATIONS)
	{

		//printf("Iteration=%d\n",iter);
		iter++;
		maxdiff = get_max_diff(a, b, p, north, south, east, west);
		com_borders(world_rank, my_column, my_row, p, a, north, south, east, west, send_buffer, recv_buffer);

		// Copy b to a
		swap_matrix(&a, &b);
		iteration++;
	}
	// tend = get_clock();
	tend = MPI_Wtime();
	ttotal = tend - tstart;

	// get_results(world_rank, n, a, b, result);

	print_grid(a, 0, p);
	// Output final grid
	printf("i'm number [%d]\n", world_rank);
	if (world_rank == 0)
	{
		printf("Final grid:\n");
		// print_array_double(world_rank, north, p, "north");
		// print_array_double(world_rank, east, p, "east");
		// print_array_double(world_rank, south, p, "south");
		// print_array_double(world_rank, west, p, "west");
		print_grid(result, 0, n);
	}
	// else
	// {
	// 	printf("i'm number [%d]\n", world_rank);
	// }
	printf("------------:\n");

	// Results
	printf("Results:\n");
	printf("Iterations=%d\n", iteration);
	printf("Tolerance=%12.10lf\n", maxdiff);
	printf("Running time=%12.10lf\n", ttotal);

	free_matrix(a, p);
	free_matrix(b, p);

	if (world_rank == 0)
	{
		free_matrix(result, n);
	}

	MPI_Finalize();

	return 0;
}
