# TrafficManagementSystem

## Software Function

The product deliverable is an intelligent traffic management system that employs
machine learning methods and the artificial bee colony (ABC) algorithm to
control and manage road traffic. Our implementation for this product consisted
of three main parts. Firstly, the machine learning model placed on sensors
(cameras) to detect vehicles. Secondly, the modification of the ABC algorithm
to work with parameters from the sensors. Finally, the integration of these two
with a set of Raspberry Pis. Ideally, this setup operates locally at one
junction, but also in conjunction with similar installations, over a
network—each at different junctions/traffic lights, working together in for
well-ordered traffic management in a locality. However, putting aside ideal
states, this project focuses on a proper setup operating at a three-way
T-junction.



## Vehicle Detection Model

The setup needs vision, and a camera works for that, it also requires some way to
process what its vision captures and recognize the number and types of vehicles
(or other entities) on the road. For this processing, we rely on an
object-detection machine learning (ML) model. To create this ML model, we
employed the use of three primary technologies. Firstly, the programming language,
Python, through the use of anaconda for package and environment management.
Next, LabelImg was an application used to label all vehicles in images obtained
from the cityscapes vehicle dataset for training and testing. And finally, at the core of
this task, TensorFlow was employed to train an object-detection model against
our vehicle image dataset. These various technologies were utilized due to
their ease-of-use, reliable documentation, and the previous experience some
group members had with it.



## Artificial Bee Colony Algorithm

The second part of our software implementation involves the use of the Artificial
Bee Colony (ABC) algorithm. This algorithm, proposed by Derviş Karaboğa
in 2005, is an optimization technique that employs, population-based search
algorithm, built of the nature of honeybees. Three major components of the
algorithm arise from its likeness to honeybee swarms, namely, food sources,
employed foragers, and unemployed foragers (Karaboga, 2005). Employed
foragers are bees "employed" at particular food sources – the
number of employed bees equals the number of food sources – on which they
report its distance and direction from the nest, and its profitability with a
proportional probability (Karaboga, 2005). Unemployed foragers consist
of two bees, first are scouts who search the environment for food
sources, and onlookers which wait in the nest and establish one reliable
food source based on what employed foragers report (Karaboga, 2005). Employed
foragers bring back two types of feedback, positive feedback or negative
feedback (Karaboga, 2005). Feedback is positive when the nectar found at a food
source increases, and the number of onlookers visiting that source also
increases. On the other hand, feedback is negative when foragers abandon a food
source (Karaboga, 2005). 



The details on how we achieve a functioning solution and the relevant data structures
in this process, come in the following comments section.



## Comments

### How it is Used

To apply the algorithm to our particular case of traffic management, we identified
the roads at junctions as sources and the number of cars as the nectar/food
from these sources. According to Karaboga (2005), the ABC algorithm is outlined
in the steps below:


!(Imgur)[https://imgur.com/w8ngOov]


From the pseudo-code above, scouts are sent to identify the food sources, employed
bees, then move onto these food sources, and the probability value for each
source is calculated. However, if this value is deemed low enough, the source
is abandoned, and it is no longer considered viable by onlookers. Next,
employed foragers turned scouts look for more food sources. The best food
source found so far is memorized, and this process described above is repeated
until the algorithm hones down on the best food source.

We aimed to modify the algorithm to suit our implementation and its constraints.
In this attempt, we realized a few issues arising from the integration. One
such problem was the parameter-narrow nature of our setup – only a few major
variables are used in our system; the count for regular cars and vehicles with
special permissions. As such, some parts of the ABC algorithm seemed overly
complicated and unneeded. Firstly, since we have pre-established all of our
parameters of known and unchanging food sources (roads), through the ML model,
the initialization method in the ABC code implementation is unneeded. Also,
employed bees are not needed to gather info from food sources (roads) when all
information is known beforehand. So, through careful consideration, we
identified what parts of the ABC algorithm/code are essential to our system,
and these were, calculating Fitness, calculating probability values,
and memorizing the best source (most traffic-packed road). Apart from
these two methods, we found no others that proved more or equally essential. This
conclusion was of no surprise, and the Artificial Bee Colony algorithm is a reasonably
sophisticated optimization technique. And the problem of determining which
roads have the most traffic, on such a small scale as we explore in this
project, is a problem with a solution too clear to require intense composite
iterative procedures. Maps store the vehicle
detection parameters for each image processed by the neural network. These maps
are later used by the modified ABC algorithm to decide which road's traffic
needs to move first. Maps (or dictionaries, as they are more commonly known in Python)
are an appropriate and vital data structure in this process as they allow for speedy
lookups to be made which is essential for a system like this which needs to
make decisions in real-time. This point is echoed by that fact that the
underlying neural network of our image detection model also uses a map
structure which allows it to make its detections quickly.


## Our Application of the ABC Algorithm

Utilizing the three essential components identified earlier from the ABC algorithm, we
came up with our modification. It works simply, it takes in data in the format
shown below, and for every source/road a fitness value – an indication of the
level of traffic – is calculated, and a probability is determined from these
values to settle on which road's traffic should be freed.

A glimpse of the code reveals the calculate_fitness
function, and it is clear it uses some sort of accumulation of a source's
(road) parameters. It works in the following way. Firstly, our accumulator
begins at zero, and as we iterate over the road's parameters ({'cars': 10, 'special_permission_vehicles': 1, 'pedestrians': 0}), the accumulator, is built up differently on each,
because they have a different effect on how urgent it is to free traffic on a
particular road.


### Effect of Regular cars on Fitness

On the iteration with the 'cars' parameter, we aim to increase the fitness value.
However, in the case of an emergency, with an ambulance on one road, we don't
want a system which says it is more critical to free 50 cars from traffic over
an ambulance on another road. So, to solve this, we use an exponential function
() on the number of cars to accumulate the fitness
value. This decreases the significance of regular cars, the more they are. It
is important to note that this diminishing marginal significance effect
only takes place when a vehicle such as an ambulance is detected. Otherwise,
the impact of the exponential function is later reversed.


### Effect of Special Permission Vehicles on Fitness

When an ambulance or another vehicle of similar priority is detected, while the diminishing
marginal significance effect takes place on regular cars, the opposite
happens for high priority vehicle. The exponential function  is used to increase
the fitness value exponentially.


### Effect of Pedestrians on Fitness

On the parameter 'pedestrians', a road has its fitness
value reduced to zero, this translates to a red light being shown on a road
when traffic lights are switching, and a pedestrian is detected.


## Technology and Justification

Many of the critical technologies for this project have already been discussed at various points
earlier in this document (i.e. LabelImg, Python, the ABC algorithm etc.) hence
this final section highlights technologies that have yet to be mentioned but
still have a relatively important role to play.


## RaspberryPi Integration

***This part of the system has not been implemented due to challenges faced with acquiring a RaspberryPi. However, below is a description of how this code repo will work with a RaspberryPi for a traffic
management system prototype.***

As this project aims to prototype an intelligent traffic management system, we
required a low-cost, credit-card sized computer that can connect to all various
sensors and devices. For this, there was no better option than the Raspberry
Pi. For lack of the better alternative – hardware – we were left to resort to
an online simulator of our Raspberry Pi demo. Integrating a Raspberry Pi, a
camera sensor, a vehicle detection model, and the ABC algorithm was the main
focus of this project. Successfully integration was to have the system working
as such:

- One Raspberry Pi, regarded as the Queen, is equipped with the ABC algorithm that
accepts parameters from three other Raspberry Pi's.

- The three other Raspberry Pi's are workers. They are equipped with a vehicle detection model to
process the number and type of vehicles captured in a connected camera. This
info is utilized by the Queen Raspberry Pi to use in making a periodical
decision for which traffic lane should move and which should not.
