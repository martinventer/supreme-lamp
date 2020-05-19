# supreme-lamp
Tool for Gareth that will assist in sampling from a hexagonal grid

## ToDo
* Generate a hexagonal grid
    * Element ID, element Coord list
* Assign a grid number to each grid point
* Plot the grid with the grid numbers
* identify adjacency of elements

## Algorithm
1. Create a grid in hexagonal coordinate
    1. Plot the grid
2. Create a network graph for the grid 
    1. Plot the network
3. Create an active subset
    1. Test node adjacency 
    2. Plot the active subnet
4. Network tests
    1. Test whether network is fully connected
    2. Test whether two networks are isomprphs
    3. Test whether two networks have identical shapes
5. Generate a list of all possible combinations 
6. Compare many markov subsets and sore the isomorphs in a dictionary



## combinatorial problem
of 19 choose 1 = 19
of 19 choose 2 = 342
of 19 choose 3 = 5814
of 19 choose 4 = 93024
of 19 choose 5 = 1395360
of 19 choose 6 = 19535040
    