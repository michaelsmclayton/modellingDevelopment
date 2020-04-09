# Genetic evolution of trees using L-systems

This project applies genetic programming to evolve tree structures (created using L-systems) which gradually develop over many generations to spread their leaves widely (emulating their natural need to maximise the light they can absorb)

<img src="./exampleGifs/TreeEvolution.gif"/>

Lindenmayer systems (or L-systems) are a method of generating fractal structures. The algorithm starts with an initial axiom (which is a single string). This string contains letters, some of which will be associated with a given transformation (or production) function. For example, the letter 'F' might invoke a function which returns 'FA', while the letter 'A' invokes a function that returns 'XF'. When a letter is not associated with a given function, the same letter is returned unchanged. If we apply these functions iteratively, we will see the following patterns emerge:<br>

Axiom: F<br>
Iteration 1: FA<br>
Iteration 2: FAXF<br>
Iteration 3: FAXFXFA<br>
Iteration 4: FAXFXFAXFAXF<br>
...<br>
<br>
By applying such simple rules iteratively, complex strings of characters can be formed. When the consistent characters of these strings are also associated with certain graphical operations (e.g. using Turtle graphics), L-systems can be used to create fractal shapes and structures. When branching is allowed, they can easily be used to create plant structures that emulate those seen in nature.

### Parametric L-system

In this example, a parametric L-system is used in which given characters are also linked with parameters. For example, the character 'F' (which brings about a forward movement in turtle graphics) has a parameter 's' (which determines the length of the forward motion). The main transformation converts all instances of the letter 'X', according to the following logic:

"X(s)" ---><br>
    s = s * RATE<br>
    if s < MIN: "F(s) [ +(ANGLE1) X(s) ] [ -(ANGLE2) X(s)] F(s) X(s)"<br>
    else: 'E'<br>

#### Graphics rules
F(s) = move turtle forward by length, s<br>
+(theta) = rotate turtle left by angle, theta<br>
-(theta) = rotate turtle right by angle, theta<br>
[ = create new branching point<br>
] = end branching point (and return turtle to start of current branch)<br>
E = draw a leaf<br>

#### Parameters
RATE = scalar value used to change the value of s every time X is detected (usually <1 to allow shortening of pen movements when tree structures are drawn)<br>
MIN = a leaf is drawn when branch length goes below this value<br>
ANGLE1 = the angle with which branches on the left side of the tree deviate from the central trunk<br>
ANGLE2 = the angle with which branches on the right side of the tree deviate from the trunk<br>

### Genetic evolution

As detailed above, the shapes produced by this parametric L-system depend on the choice of input parameters. In some sense, these parameters can be thought of as the 'chromosome' of the tree, containing compressed, genetic-like information of how to generate a given tree structure. With these parameters, this code uses genetic algorithms to slowly evolve tree structures to maximise a given fitness function. Here, fitness is determined by the number of leafs that recieve direct sunlight from above. The population size is kept constant at 20, and at every generation, parents are chosen to reproduce using a tournament method (implemented in DEAP). Children are then produced by randomly combining the parameters of selected parents, creating a new generation.

Together, when performed iteratively over many generations, this process leads to the selection and slow evolution of L-system parameters which grow tree structures that spread their leaves widely, maximising the amount of sunlight that the tree can absorb. Examples of how the fittest individual in each generation develops over time can be seen in the animation above.

### Description of code

- ./TreeLSystem.py<br><br>
This script specifies an L-system class which, when given the necessary parameters, can generate and plot a tree structure. It's main function run() returns the fitness of the created structure (i.e. the amount of sunlight which it collects from above)

- ./parametricTreeEvolution.py<br><br>
Specifies the starting parameters for the initial L-system population, and the necessary code to perform the evolution of these systems over many generations. The evolutionary computation is done using DEAP.


### Dependencies
- deap
- turtle
- numpy
- matplotlib.pyplot