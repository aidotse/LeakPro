/*
 * BCJR decoder for Group testing with
 * MAP decoding on Trellis
 * based on the work
 * "Optimum Detection of Defective Elements in Non-Adaptive Group Testing"
 * G. Liva, E. Paolini and M. Chiani, 2021.
 */
/* C Libraries */

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>

#define LUT_SIZE 128
#define LUT_RANGE 5.5f
#define LUT_FACTOR (LUT_SIZE / LUT_RANGE)
#define LUT_STEP (LUT_RANGE / LUT_SIZE)
#define LOG_ZERO (-1E6)
#define NO_STATE (-1)
#define VALID_STATE(s) ((s) >= 0)

/* Needed global variables */
uint8_t **Hmat;
int K, N;
double Channel_model[2][2]; // Binary channel model P(i/j) =Channel_model[i][j]
// Trellis description
static int maxstates;
static int num_of_from_states;
static int ***tostate;
static int ***fromstate;
static int *numstates;
// MAP decoding arrays
static double **log_alpha;
static double **log_beta;
static double lut_fc[LUT_SIZE];

static uint8_t **malloc2d_i8(size_t nx, size_t ny)
{
	size_t i;
	uint8_t **p = malloc(nx * sizeof(uint8_t *));
	p[0] = malloc(nx * ny * sizeof(uint8_t));
	for (i = 1; i < nx; i++)
		p[i] = p[i - 1] + ny;
	return p;
}

static void free2d_i8(uint8_t **p)
{
	free(p[0]);
	free(p);
}

static double **malloc2d_d(size_t nx, size_t ny)
{
	size_t i;
	double **p = malloc(nx * sizeof(double *));
	p[0] = malloc(nx * ny * sizeof(double));
	for (i = 1; i < nx; i++)
		p[i] = p[i - 1] + ny;
	return p;
}

static void free2d_d(double **p)
{
	free(p[0]);
	free(p);
}

/* Allocate/free memory for 3D arrays */

static int ***malloc3d_i(size_t nx, size_t ny, size_t nz)
{
	size_t i;
	int ***p = malloc(nx * sizeof(int **));
	p[0] = malloc(nx * ny * sizeof(int *));
	p[0][0] = malloc(nx * ny * nz * sizeof(int));
	for (i = 1; i < nx; i++)
		p[i] = p[i - 1] + ny;
	for (i = 1; i < (nx * ny); i++)
		p[0][i] = p[0][i - 1] + nz;
	return p;
}

static void free3d_i(int ***p)
{
	free(p[0][0]);
	free(p[0]);
	free(p);
}

/* initialize Hmat */

void initializeHmat(const uint8_t *H, uint64_t r, uint64_t n)
{
	Hmat = malloc2d_i8(r, n);
	for (size_t i = 0; i < r; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			Hmat[i][j] = H[j + i * n];
		}
	}
}

void init_codec()
{
	// setup the map decoder
	log_alpha = malloc2d_d(N + 1, maxstates);
	log_beta = malloc2d_d(N + 1, maxstates);

	// build the max-star look up table
	double x = 0.0;
	for (int i = 0; i < LUT_SIZE; i++)
	{
		lut_fc[i] = log(1.0 + exp(-x));
		x += LUT_STEP;
	}
}

static int bin2dec(int m, int *bits)
{
	int res = 0;
	for (int i = 0; i < m; i++)
		res += bits[i] * pow(2, i);
	return res;
}

static void dec2bin(int dec, int m, int *bits)
{
	int i;
	for (i = m - 1; i >= 0; i--)
		bits[i] = (dec >> i) & 1;
}

static double noisy_test_outcome(int state, const uint8_t *test_vec)
{
	int *state_vec = (int *)malloc((N - K) * sizeof(int));
	dec2bin(state, N - K, state_vec);
	double probability = 1;
	for (int i = 0; i < N - K; i++)
	{
		probability *= Channel_model[test_vec[i]][state_vec[i]];
	}
	free(state_vec);
	return probability;
}

/* Build Trellis*/
static void build_trellis()
{
	int t, i, curstate, nextstate, num_deleted_states;
	int *syndrome, *deleted_states, *states_labels;
	int total_edges, *num_edges;

	maxstates = 1 << (N - K);
	num_of_from_states = maxstates + 1;
	numstates = malloc((N + 1) * sizeof(int));
	tostate = malloc3d_i(N + 1, maxstates, 2);
	fromstate = malloc3d_i(N + 1, maxstates, num_of_from_states);
	/* build the full trellis */
	for (t = 0; t <= N; t++)
	{
		numstates[t] = 0;
		for (curstate = 0; curstate < maxstates; curstate++)
		{
			tostate[t][curstate][0] = NO_STATE;
			tostate[t][curstate][1] = NO_STATE;
			for (int state_it = 0; state_it < num_of_from_states; state_it++)
				fromstate[t][curstate][state_it] = NO_STATE;
		}
	}
	syndrome = malloc((N - K) * sizeof(int));
	fromstate[0][0][0] = 0;
	for (t = 0; t < N; t++)
	{
		for (curstate = 0; curstate < maxstates; curstate++)
		{
			if (VALID_STATE(fromstate[t][curstate][0]) || VALID_STATE(fromstate[t][curstate][1]))
			{
				dec2bin(curstate, N - K, syndrome);
				nextstate = curstate;
				tostate[t][curstate][0] = nextstate;
				fromstate[t + 1][nextstate][0] = curstate; // Seems nothing to be changed here
				for (i = 0; i < N - K; i++)
					syndrome[i] |= Hmat[i][t];
				nextstate = bin2dec(N - K, syndrome);
				tostate[t][curstate][1] = nextstate;
				for (int state_it = 1; state_it < num_of_from_states; state_it++)
				{
					if (!VALID_STATE(fromstate[t + 1][nextstate][state_it]))
					{
						fromstate[t + 1][nextstate][state_it] = curstate;
						break;
					}
				}
			}
		}
	}

	fromstate[0][0][0] = NO_STATE;
	free(syndrome);

	/* remove paths that do not end in the zero state -- Cannot do this for group testing  */
	numstates[0] = 1;
	/* This might cause a bug for some stupid parity-check matrix*/
	numstates[N] = maxstates; // Check if this it always holds!
	deleted_states = malloc((maxstates - 1) * sizeof(int));
	num_deleted_states = maxstates - 1;
	for (i = 1; i < maxstates; i++)
		deleted_states[i - 1] = i;
	for (t = N - 1; t > 0; t--)
	{
		numstates[t] = maxstates;
		for (curstate = 0; curstate < maxstates; curstate++)
		{
			if (tostate[t][curstate][0] == NO_STATE && tostate[t][curstate][1] == NO_STATE)
				numstates[t]--; // If from a state s in depth 0 to N-1, no edges go out, this state is not a Valid state!
		}
		num_deleted_states = maxstates - numstates[t];
		i = 0;
		for (curstate = 0; curstate < maxstates; curstate++)
			if (tostate[t][curstate][0] == NO_STATE && tostate[t][curstate][1] == NO_STATE)
				deleted_states[i++] = curstate;
	}
	free(deleted_states);

	/* final pass to update the states labels (from 0 to numstates[t]-1) */
	states_labels = malloc(maxstates * sizeof(int));
	for (t = 1; t < N; t++)
	{
		int oldlabel, newlabel, nextlabel = 1;
		states_labels[0] = 0;
		for (curstate = 1; curstate < maxstates; curstate++)
		{
			if (VALID_STATE(tostate[t][curstate][0]) || VALID_STATE(tostate[t][curstate][1]))
				states_labels[nextlabel++] = curstate;
		}
		if (nextlabel != numstates[t])
		{ // makes sense
			fprintf(stderr, "An error has occured during the construction of the trellis\n");
			exit(200);
		}
		for (newlabel = 0; newlabel < numstates[t]; newlabel++)
		{
			oldlabel = states_labels[newlabel];
			if (newlabel != oldlabel)
			{
				tostate[t][newlabel][0] = tostate[t][oldlabel][0];
				if (VALID_STATE(tostate[t][newlabel][0]))
					fromstate[t + 1][tostate[t][newlabel][0]][0] = newlabel;
				tostate[t][newlabel][1] = tostate[t][oldlabel][1];
				if (VALID_STATE(tostate[t][newlabel][1]))
				{
					int state_sc = -1;
					for (int state_it = 1; state_it < num_of_from_states; state_it++) // find the old label place
					{
						if (fromstate[t + 1][tostate[t][newlabel][1]][state_it] == oldlabel)
						{
							state_sc = state_it;
							fromstate[t + 1][tostate[t][newlabel][1]][state_sc] = newlabel;
							break;
						}
					}
					if (state_sc == -1)
					{
						printf("state_sc should have been updated!\n");
						exit(-2);
					}
				}
				tostate[t][oldlabel][0] = NO_STATE;
				tostate[t][oldlabel][1] = NO_STATE;
				fromstate[t][newlabel][0] = fromstate[t][oldlabel][0];
				if (VALID_STATE(fromstate[t][newlabel][0]))
					tostate[t - 1][fromstate[t][newlabel][0]][0] = newlabel;
				fromstate[t][oldlabel][0] = NO_STATE;
				for (int state_it = 1; state_it < num_of_from_states; state_it++)
				{
					fromstate[t][newlabel][state_it] = fromstate[t][oldlabel][state_it];
					if (VALID_STATE(fromstate[t][newlabel][state_it]))
						tostate[t - 1][fromstate[t][newlabel][state_it]][1] = newlabel;
					fromstate[t][oldlabel][state_it] = NO_STATE;
				}
			}
		}
	}
	free(states_labels);

	/* count the total number of edges */
	total_edges = 0;
	num_edges = malloc(N * sizeof(int));
	for (t = 0; t < N; t++)
	{
		num_edges[t] = 0;
		for (curstate = 0; curstate < numstates[t]; curstate++)
		{
			if (VALID_STATE(tostate[t][curstate][0]))
				num_edges[t]++;
			if (VALID_STATE(tostate[t][curstate][1]))
				num_edges[t]++;
		}
		total_edges += num_edges[t];
	}
	free(num_edges);
}
/* End of trellis building  */
static double maxstar(double x, double y)
{
	int k;
	double z;
	z = (x > y) ? x : y;
	k = (int)fabs((x - y) * LUT_FACTOR);
	if (k < LUT_SIZE)
		z += lut_fc[k];
	return z;
}

/* Beginning of map_decoder */
/* Trellis-based MAP decoder
 *
 * Input:  - llri = input LLRs on code bits
 * Output: - llro = extrinsic LLRs on code bits
 *         - dec = hard decisions on code bits */

void map_decoder(const double *llri, double *llro, uint8_t *dec, double threshold_dec, const uint8_t *test_vec)
{
	int t, curstate, prevstate, nextstate;
	static double log_app[2];

	/* forward recursion */
	log_alpha[0][0] = 0.0;
	for (t = 1; t < N; t++)
	{
		double normfac = LOG_ZERO;
		for (curstate = 0; curstate < numstates[t]; curstate++)
		{
			log_alpha[t][curstate] = LOG_ZERO;
			prevstate = fromstate[t][curstate][0];
			if (VALID_STATE(prevstate))
				log_alpha[t][curstate] = maxstar(log_alpha[t][curstate], log_alpha[t - 1][prevstate] + llri[t - 1]);
			for (int state_it = 1; state_it < num_of_from_states; state_it++)
			{
				prevstate = fromstate[t][curstate][state_it];
				if (VALID_STATE(prevstate))
					log_alpha[t][curstate] = maxstar(log_alpha[t][curstate], log_alpha[t - 1][prevstate] + 0);
			}
			normfac = maxstar(normfac, log_alpha[t][curstate]);
		}
		for (curstate = 0; curstate < numstates[t]; curstate++)
			log_alpha[t][curstate] -= normfac;
	}

	/* backward recursion */
	for (int statusija = 0; statusija < maxstates; statusija++)
	{
		double temp_testi = noisy_test_outcome(statusija, test_vec);
		if (temp_testi > LOG_ZERO)
			log_beta[N][statusija] = log(temp_testi);
		else if (temp_testi < 0)
		{
			printf("Problem with probabilities!!");
			exit(-5);
		}
		else
			log_beta[N][statusija] = LOG_ZERO;
	}
	for (t = N - 1; t > 0; t--)
	{
		double normfac = LOG_ZERO;
		for (curstate = 0; curstate < numstates[t]; curstate++)
		{
			log_beta[t][curstate] = LOG_ZERO;
			nextstate = tostate[t][curstate][0];
			if (VALID_STATE(nextstate))
				log_beta[t][curstate] = maxstar(log_beta[t][curstate], llri[t - 1] + log_beta[t + 1][nextstate]);
			nextstate = tostate[t][curstate][1];
			if (VALID_STATE(nextstate))
				log_beta[t][curstate] = maxstar(log_beta[t][curstate], 0 + log_beta[t + 1][nextstate]);
			normfac = maxstar(normfac, log_beta[t][curstate]);
		}
		for (curstate = 0; curstate < numstates[t]; curstate++)
			log_beta[t][curstate] -= normfac;
	}

	/* LLRs on coded bits (merge the two recursions) */
	for (t = 0; t < N; t++)
	{
		log_app[0] = log_app[1] = LOG_ZERO;
		for (curstate = 0; curstate < numstates[t]; curstate++)
		{
			nextstate = tostate[t][curstate][0];
			if (VALID_STATE(nextstate))
				log_app[0] = maxstar(log_app[0], log_alpha[t][curstate] + llri[t] + log_beta[t + 1][nextstate]);
			nextstate = tostate[t][curstate][1];
			if (VALID_STATE(nextstate))
				log_app[1] = maxstar(log_app[1], log_alpha[t][curstate] + 0 + log_beta[t + 1][nextstate]);
		}
		llro[t] = log_app[0] - log_app[1];
		dec[t] = (llro[t] > threshold_dec) ? 0 : 1;
	}
}
/* End of map_decoder*/

/* Mexfunction Interface */
void BCJR(const uint8_t *H, const double *LLRinput, const uint8_t *test_vec, const double *Chanelmatrix, const double threshold_dec, const int no_N, const int r, double *LLRO, uint8_t *DEC)
{
	N = no_N;
	K = N - r;
	Channel_model[0][0] = Chanelmatrix[0];
	Channel_model[0][1] = Chanelmatrix[1];
	Channel_model[1][0] = Chanelmatrix[2];
	Channel_model[1][1] = Chanelmatrix[3];
	initializeHmat(H, r, N);
	build_trellis();
	init_codec();
	free2d_i8(Hmat);

	map_decoder(LLRinput, LLRO, DEC, threshold_dec, test_vec);

	free(numstates);
	free2d_d(log_alpha);
	free2d_d(log_beta);
	free3d_i(tostate);
	free3d_i(fromstate);
}