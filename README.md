## Example Tree Visualizations

Each node in a tree has a type, represented by a token such as `+` or `abs`.
Each token represents a function/operator to be applied to its children.

Each node also has a `ord` (short for order) property, which communicates
which child index a particular node is, if the index matters. For the root node
and for nodes that describe commutative functions, the `ord` value is `-1`,
which means "unordered."

Finally, in these visualizations, there is an index value that tells what index
in the adjacency matrix each node is. This isn't important except for as a
sanity check for the tree data generator.

![addition.pdf](/viz/addition.pdf)
![division_abs.pdf](/viz/division_abs.pdf)
![multiplication.pdf](/viz/multiplication.pdf)
![mult_subtract.pdf](/viz/mult_subtract.pdf)
