# crystal_sim
- Simulating serial electron diffraction patterns to train a neural network for crystal structure determination (classification and regression).
- Using the Materials Project Database, I am able to process large amounts of data from various classes of crystal systems. 
- The input features are multiple diffraction patterns from various orientations (zone axes). The labels for classification can be the space group number, the Bravais Lattice type or the crystal system. The labels for regression are the unit cell parameters $(a, b, c, \alpha, \beta, \gamma)$.