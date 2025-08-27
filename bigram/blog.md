# Building a Tiny Bigram Name Generator in C

After watching Andrej Karpathy's video on building *makemore*, I wanted to try something similar. But instead of Python, I chose C, since I'm doing a series on building things from scratch in C.

The goal was to see, end-to-end, how a very simple character-based model can synthesize plausible baby names. No heavy libraries or frameworks - just C, a text file of names, and some loops.

This post walks through the process of building a bigram model: reading data, counting transitions, normalizing into probabilities, sampling, and generating names. I'll also note some tradeoffs, pitfalls, and ways to extend the model.

## Background

Suppose we have a dataset of names. We want to generate new names that *feel* like the ones in our dataset. One of the simplest approaches is a **bigram model**.

### What is a bigram?

A bigram is a very simple model that looks only at pairs of characters. To generate text, it predicts the next character using only the current one. It completely ignores longer context.

`"anna"` -> `".anna."` -> `[(.,a), (a,n), (n,n), (n,a), (a,.)]`

In this setup:

* We use the alphabet `a–z` plus one special token `.` that represents both the start and end of a name.
* The model learns how often each character follows another, stored in a **27×27 transition matrix**.
* Probabilities come from normalizing counts row by row.

Generation is a simple Markov chain: start at `.`, repeatedly sample the next character using probabilities, and stop when `.` is sampled again.

## Dataset and tokens

Our dataset is a text file where each line is a name. While reading, we wrap every name with the start/end token `.` so the model can learn boundaries. All names are assumed to be in lowercase.

```txt
alice
kevin
...
```
becomes  
`.alice.`  
`.kevin.`

Two practical simplifications:

* Fixed constants for maximum names and lengths (easy memory management).
* A single token `.` for both start and end boundaries, which keeps the vocabulary at exactly 27 tokens.

## Counting transitions

Next, we walk through every name once and count character pairs. This fills a 27×27 integer matrix.

* `.` to letter → captures how names begin
* letter to `.` → captures how names end
* letter to letter → captures character sequences inside names

![heatmap](https://github.com/aasishraj/makemore-c/blob/main/bigram/public/heatmap.png)
*The heatmap above shows character transition frequencies.*

This pass is linear in the total number of characters.

## Converting counts to probabilities

Each row of the matrix is normalized so that probabilities sum to 1. For example, if after `a` the dataset has 60 occurrences of `n`, 30 of `l`, and 10 of `.` then:

$P(n \mid a) = 0.6$  
$P(l \mid a) = 0.3$  
$P(\text{.} \mid a) = 0.1$

We do not apply smoothing here, so unseen bigrams have zero probability. For very small datasets, add-one (Laplace) smoothing helps avoid dead ends.

## Sampling from the model

If we always picked the most probable next character, we would get the same names every run. To keep it stochastic, we **sample** from the probability distribution instead.

This is essentially drawing one sample from a categorical distribution at each step.

Implementing **multinomial sampling** was the hardest piece in the entire code to understand because in the video, Karpathy Sensei uses `torch.multinomial` which is roughly implemented in C as given below.

### What problem are we solving?

Imagine you have a die-like object, but instead of 6 faces, it has **k sides**, and each side has a different probability of being chosen. That's a **categorical distribution**.

Now, instead of rolling this dice once, imagine rolling it **n times** and keeping track of how many times each side came up. That's the **multinomial distribution**.

For example:

* Probabilities = `[0.5, 0.3, 0.2]` (like a loaded dice with 3 sides)
* If `n = 10` rolls, one possible outcome could be `[4, 3, 3]` -> side 1 appeared 4 times, side 2 appeared 3 times, side 3 appeared 3 times.

The function `multinomial_sample` is a way to simulate that.

### How does this function do it?
```c
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
```

Let's walk through the logic:

1. **uniform\_rand()**
   Generates a random number between 0 and 1. Think of it as spinning a fair roulette wheel.

2. **binomial\_sample(n, p)**
   This asks: “If I flip a biased coin `n` times, where the chance of heads is `p`, how many heads will I get?”

   * For each flip, it uses `uniform_rand()` to decide if it's a success (head) or not.
   * After all flips, it returns the number of successes.

3. **multinomial\_sample(n, p, k, out)**
   Here's the clever part:

   * You want to distribute `n` trials across `k` categories according to probabilities `p[0], p[1], …, p[k-1]`.
   * Start with all `n` trials unassigned.
   * For the first category:

     * Figure out how many of those `n` trials should go here. That's just a binomial draw with probability proportional to `p[0]`.
   * Subtract that count from `remaining`.
   * Move to the next category and repeat, but now probabilities are rescaled to reflect the “still unassigned” part.
   * The last category automatically gets whatever trials remain.

## Name generation

To generate a name:

1. Start with the `.` token.
2. Sample the next character using probabilities.
3. Print it if it's a letter.
4. Stop when another `.` is drawn.

Repeat this process to produce as many names as desired.

## Build and run

From the repository root:

```bash
cd bigram
gcc bigram.c -o bigram && ./bigram
```

With a fixed random seed, you'll always get the same sequence of names across runs.

## Easy extensions

* Add Laplace smoothing: `(count + 1) / (row_sum + V)` where `V = 27`.
* Replace multinomial sampling with a simpler cumulative probability scan.
* Visualize the transition matrix as a heatmap for better intuition.

## Closing thoughts

This small project is a complete pipeline from raw text to generated names. The point isn't to compete with modern models, but to build intuition about the basics: tokenization, counting, normalization, and sampling.

Once you understand these steps, scaling up to n-grams, neural networks, or better tokenization becomes much less mysterious.

Thanks for reading.

### References

[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) by Andrej Karpathy