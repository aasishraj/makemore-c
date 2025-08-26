#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#define MAX_NAMES 32033
#define MAX_NAMES_LEN 20
#define TOKENS 27

/*
Read names from file adding start and end tokens at file operations
*/
char **read_names(int *count) {
    FILE *file = fopen("names.txt", "r");
    if (file == NULL) {
        perror("Error opening file!");
        exit(1);
    }

    char **names = malloc(MAX_NAMES * sizeof(char *));
    if (names == NULL) {
        perror("Memory allocation for names array failed!");
        exit(1);
    }

    char buffer[MAX_NAMES_LEN];
    *count = 0;

    while (fgets(buffer, sizeof(buffer), file) != NULL && *count < MAX_NAMES) {
        buffer[strcspn(buffer, "\n")] = 0;  // strip newline

        names[*count] = malloc(strlen(buffer) + 3);
        if (names[*count] == NULL) {
            perror("Mem allocation failed!");
            exit(1);
        }

        snprintf(names[*count], MAX_NAMES_LEN + 3, ".%s.", buffer);
        (*count)++;
    }

    fclose(file);
    return names;
}

/*
Distribution matrix maker with 27 token characters (a to z, '.')
*/
int ** construct_distribution_matrix(char **names, int count) {
    // create matrix    
    int **distribution_matrix = (int **)malloc(TOKENS * sizeof(int *));
    if (distribution_matrix == NULL) {
        perror("malloc failed");
        exit(1);
    }

    for (int i = 0; i < TOKENS; ++i) {
        distribution_matrix[i] = (int *)calloc(TOKENS, sizeof(int));
        if (distribution_matrix[i] == NULL) {
            perror("calloc failed");
            exit(1);
        }
    }

    for (int i = 0; i < count; ++i) {
        for (int j = 0, k = 1; names[i][k] != '\0'; ++j, ++k) {

            if (names[i][j] == '.')
                distribution_matrix[26][names[i][k] - 97] += 1;
            
            else if (names[i][k] == '.')
                distribution_matrix[names[i][j] - 97][26] += 1;  
      
            else 
                distribution_matrix[names[i][j] - 97][names[i][k] - 97] += 1;
        }
    }
    
    return distribution_matrix;
}

double **convert_to_probabilities(int **freq_matrix, int size) {
    double **prob_matrix = malloc(size * sizeof(double *));
    if (!prob_matrix) {
        perror("malloc failed");
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        prob_matrix[i] = malloc(size * sizeof(double));
        if (!prob_matrix[i]) {
            perror("malloc failed");
            exit(1);
        }

        // compute row sum
        long row_sum = 0;
        for (int j = 0; j < size; j++) {
            row_sum += freq_matrix[i][j];
        }

        // normalize row
        for (int j = 0; j < size; j++) {
            if (row_sum > 0)
                prob_matrix[i][j] = (double)freq_matrix[i][j] / row_sum;
            else
                prob_matrix[i][j] = 0.0;
        }
    }

    return prob_matrix;
}


// multinomial sampling helper functions

double uniform_rand() {
    return (rand() + 1.0) / (RAND_MAX + 2.0);
}

int binomial_sample(int n, double p) {
    int x = 0;
    for (int i = 0; i < n; i++) {
        if (uniform_rand() < p) {
            x++;
        }
    }
    return x;
}

/*
Does multinomial sampling over a probability distribution
*/
void multinomial_sample(int n, double *p, int k, int *out) {
    int remaining = n;
    double cumulative_p = 1.0;

    for (int i = 0; i < k - 1; i++) {
        double pi = p[i] / cumulative_p;
        int xi = binomial_sample(remaining, pi);
        out[i] = xi;
        remaining -= xi;
        cumulative_p -= p[i];
    }
    out[k - 1] = remaining;
}

/*
Generate a name
*/
void generate_name(double **prob_matrix) {
    // take multinomial sample for each character token
    
    int current = 26;  // start token index for '.'
    while (1) {
        int sample_out[27];
        multinomial_sample(1, prob_matrix[current], TOKENS, sample_out);

        // find which token got picked
        int next = -1;
        for (int j = 0; j < TOKENS; j++) {
            if (sample_out[j] > 0) {
                next = j;
                break;
            }
        }

        if (next == 26) { // end token
            printf("\n");
            break;
        } else {
            printf("%c", 'a' + next);
            current = next;
        }
    }
}


void collect_garbage(int **freq, char **names, int names_count, double **prob) {
    for (int i = 0; i < TOKENS; i++) free(freq[i]);
    free(freq);

    for (int i = 0; i < names_count; i++) free(names[i]);
    free(names);

    for (int i = 0; i < TOKENS; i++) free(prob[i]);
    free(prob);
}


int main() {
    unsigned int seed = 2147483647;
    srand(seed);

    // Read names
    int count = 0;
    char **names = read_names(&count);

    // construct the frequency matrix
    int **distribution_matrix = construct_distribution_matrix(names, count);


    // convert frequency distributio to probability
    double **probability_distribution = convert_to_probabilities(distribution_matrix, TOKENS);
 
    // generate names
    int names_count = 40;
    for (int i = 0; i < names_count; i++) 
        generate_name(probability_distribution);

    collect_garbage(distribution_matrix, names, count, probability_distribution);
    
    return 0;
}
