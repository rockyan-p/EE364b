Tetrahymena thermophila, a single-cell swimming organism, exhibits seven distinct mating types. Each mating type can interact with any other except its own, a phenomenon facilitated by two surface proteins, Mta and Mtb. There are seven each of Mta(1-7) and Mtb(1-7) and the proteins are likely recognizing the "self" protein 
(Mtb ≠ Mtb, or Mtb ≠ Mta).

Google Deepmind's AlphaFold 3, the state-of-the-art diffusion based generative protein structural model, can predict pairwise interactions between any mating type proteins.

Prediction confidence is low at interfaces. We use convex optimization to extend structural predictions of the mating protein family into functional predictions of a protein binding network.  

Atomic contact probability matrices are treated as noisy adjacency matrices, with sparse contacts. With convex optimization we formulate a sparse regression that  identifies critical residues that are differently self-interacting.
These candidates are experimentally testable, letting 
Tetrahymena inform the understanding of protein computation and molecular recognition.
