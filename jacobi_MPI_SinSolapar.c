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

	// Compute border values of the submatrix
	// Formula: self + north + south + west + east

	for (i = 0; i < p; i++)
	{
		for (j = 0; j < p; j++)
		{
			//First row
			if (i == 0)
			{
				//First column
				if (j == 0)
				{
					b[i][j] = 0.2 * (a[i][j] + north[j] + a[i + 1][j] + west[i] + a[i][j + 1]);
				}
				//Last Column
				else if (j == p - 1)
				{
					b[i][j] = 0.2 * (a[i][j] + north[j] + a[i + 1][j] + a[i][j - 1] + east[i]);
				}
				else
				{
					b[i][j] = 0.2 * (a[i][j] + north[j] + a[i + 1][j] + a[i][j - 1] + a[i][j + 1]);
				}
			}
			//Last Row
			else if (i == p - 1)
			{
				//First column
				if (j == 0)
				{
					b[i][j] = 0.2 * (a[i][j] + a[i - 1][j] + south[j] + west[i] + a[i][j + 1]);
				}
				//Last Column
				else if (j == p - 1)
				{
					b[i][j] = 0.2 * (a[i][j] + a[i - 1][j] + south[j] + a[i][j - 1] + east[i]);
				}
				else
				{
					b[i][j] = 0.2 * (a[i][j] + a[i - 1][j] + south[j] + a[i][j - 1] + a[i][j + 1]);
				}
			}
			//Inner Rows
			else
			{
				//First column
				if (j == 0)
				{
					b[i][j] = 0.2 * (a[i][j] + a[i - 1][j] + a[i + 1][j] + west[i] + a[i][j + 1]);
				}
				//Last Column
				else if (j == p - 1)
				{
					b[i][j] = 0.2 * (a[i][j] + a[i - 1][j] + a[i + 1][j] + a[i][j - 1] + east[i]);
				}
				else
				{
					b[i][j] = 0.2 * (a[i][j] + a[i - 1][j] + a[i + 1][j] + a[i][j - 1] + a[i][j + 1]);
				}
			}
			
			if (fabs(b[i][j] - a[i][j]) > maxdiff)
			{
				maxdiff = fabs(b[i][j] - a[i][j]);
			}
		}
	}

	return maxdiff;
}

void com_borders(int rank, int col, int row, int p, double **a, double *north, double *south, double *east, double *west, double *send_buffer)
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
	//If im not in last row, i have to send my bottom border
	if (row != p - 1)
	{
		for (j = 0; j < p; j++)
		{
			send_buffer[j] = a[p - 1][j];
		}
		MPI_Send(send_buffer, p, MPI_DOUBLE, rank + p, 0, MPI_COMM_WORLD);
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
	//If im not in last column, I have to send my right border
	if (col != p - 1)
	{
		for (j = 0; j < p; j++)
		{
			send_buffer[j] = a[j][p - 1];
		}
		MPI_Send(send_buffer, p, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
	}

	//If im in not in the first row, ill receive a top border
	if (row != 0)
	{
		MPI_Recv(north, p, MPI_DOUBLE, rank - p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	// If im not in the last row, ill receive a bottom border
	if (row != p - 1)
	{
		MPI_Recv(south, p, MPI_DOUBLE, rank + p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	//If im not in the first column, ill receive a left border
	if (col != 0)
	{
		MPI_Recv(west, p, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	//If im not in the last column, ill receive a right border
	if (col != p - 1)
	{
		MPI_Recv(east, p, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}

void fill_res(double *rcv, int rank, int p, double **res)
{
	int i, j, iter;
	int col = (rank % p);
	int row = (int)(rank / p);

	// if (rank == 1) {
	// 	printf("Filling result of rank [%d] col [%d] and row [%d] ->\n", rank, col, row);
	// 	print_array_double(rank, rcv, p, "received");
	// }

	iter = 0;
	for (i = 0; i < p; i++)
	{
		for (j = 0; j < p; j++)
		{
			res[(row * p) + i][(col * p) + j] = rcv[iter];
			// if (rank == 1) printf("Filled result[%d][%d] with value [%6.4lf] \n", row + i, col + j, rcv[iter]);
			iter++;
		}
	}
}

void get_results(int rank, int p, double **a, double **b, double **res)
{
	int iter = 0;
	double *res_buff = malloc(sizeof(double) * (p * p));

	for (int i = 0; i < p; i++)
	{
		for (int j = 0; j < p; j++)
		{
			res_buff[iter] = a[i][j];
			iter++;
		}
	}

	// Receive results from other nodes
	if (rank == 0)
	{
		fill_res(res_buff, 0, p, res);

		// print_grid(res, 0, p * p); result ok till here

		for (int i = 1; i < p * p; i++)
		{
			MPI_Recv(res_buff, p * p, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			// printf("Received from [%d]: \n", i);
			// print_array_double(rank, res_buff, p * p, "resarr");
			fill_res(res_buff, i, p, res);
			// if (i == 1) print_grid(res, 0, p * p);
		}
	}
	else
	{
		// Send my results to root
		MPI_Send(res_buff, p * p, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}

	free(res_buff);
}

int main(int argc, char *argv[])
{
	int i, j, p, n, iteration, my_column, my_row;
	double **a, **b, **result, maxdiff, newdiff, new_ttotal_com, local_diff;
	double tstart, tend, ttotal;
	double tstart_comm, tend_comm, ttotal_comm;

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
		// print_grid(result, 0, n);
	}

	//Sides for the submatrices of each proccess
	double *north, *south, *east, *west;
	north = malloc(sizeof(double) * p);
	south = malloc(sizeof(double) * p);
	east = malloc(sizeof(double) * p);
	west = malloc(sizeof(double) * p);

	double *send_buffer = malloc(sizeof(double) * p);

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
	while (maxdiff > TOL && iteration < MAX_ITERATIONS)
	{
		local_diff = get_max_diff(a, b, p, north, south, east, west);
		// if (local_diff > 0.7 ) printf("local_diff: [%12.10lf] rank: [%d]\n", local_diff, world_rank);

		//comn time start
		tstart_comm = MPI_Wtime();
		com_borders(world_rank, my_column, my_row, p, a, north, south, east, west, send_buffer);
		//comn time end
		tend_comm = MPI_Wtime();

		// All get the maximum diff
		MPI_Allreduce(&local_diff, &newdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		maxdiff = newdiff;

		// Copy b to a
		swap_matrix(&a, &b);
		iteration++;
		ttotal_comm += tend_comm - tstart_comm;  
	}
	// tend = get_clock();
	tend = MPI_Wtime();
	ttotal = tend - tstart;

	get_results(world_rank, p, a, b, result);

	MPI_Allreduce(&ttotal_comm, &new_ttotal_com, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// printf("i'm number [%d]\n", world_rank);
	// printf("*************\n");
	// print_grid(a, 0, p);
	// printf("*************\n");

	// Output final grid
	if (world_rank == 0)
	{

		printf("Final grid:\n");
		// print_array_double(world_rank, north, p, "north");
		// print_array_double(world_rank, east, p, "east");
		// print_array_double(world_rank, south, p, "south");
		// print_array_double(world_rank, west, p, "west");
		print_grid(result, 0, n);
		printf("------------:\n");

		// Results
		printf("Results:\n");
		printf("Iterations=%d\n", iteration);
		printf("Tolerance=%12.10lf\n", maxdiff);
		printf("Running simulation time=%12.10lf\n", ttotal);
		printf("Communication time mean per node=%12.10lf\n", new_ttotal_com / n);
		printf("Step time mean=%12.10lf\n", ttotal / iteration);

		free_matrix(result, n);
	}
	// else
	// {
	// 	printf("i'm number [%d]\n", world_rank);
	// }

	free_matrix(a, p);
	free_matrix(b, p);
	free(north);
	free(south);
	free(east);
	free(west);
	free(send_buffer);

	// printf("number [%d] ended\n", world_rank);
	MPI_Finalize();

	return 0;
}
