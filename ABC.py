from math import e as exp

class abcModified():

    def __init__(self, sources: list):
        """
        Class Constructor.

        sources (parameter):  contains the variables (count and type of various vehicles) of
        all food sources (roads). Format of this parameter is as such: 
        {
            road_id_1: {'cars': 20, 'special_permission_vehicles': 0, 'pedestrians': 0}, 
            road_id_2: {'cars': 10, 'special_permission_vehicles': 1, 'pedestrians': 0},
            road_id_3: {'cars': 50, 'special_permission_vehicles': 0, 'pedestrians': 0}
        }
        """

        self.sources = sources
        self.fitness = {}
        self.probabilities = {}
        self.best_source = None
        self.special_permission_vehicle_present = False
    
    def calculate_fitness(self):
        """
        This method calculates the concentration of nectar (amount/level of traffic) at 
        each food source (road).
        This value is obtained using the exponential function.
        """
        for source in self.sources:
            fitness_val = 0
            parameters = self.sources[source]
            for parameter in parameters:
                if parameters[parameter] == 0:
                    continue

                # Increase fitness if special permission vehicles are used
                if parameter == 'special_permission_vehicles':
                    fitness_val += exp ** parameters[parameter]
                    self.special_permission_vehicle_present = True

                # Decrease fitness if pedestrians are on the road; cars would have to stop
                elif parameter == 'pedestrians':
                    fitness_val = 0 if parameters[parameter] > 0 else fitness_val
                
                # Increase fitness, by smaller amounts if more regular cars are on the road.
                # This allows the presence of one special permission vehicle to dominate
                # the presence of many other regular vehicles.
                else:
                    fitness_val += exp ** (-parameters[parameter])
            self.fitness[source] = fitness_val


    def calculate_probabilities(self):
        """
        This method uses the fitness values of food sources to determine which has the 
        highest probability to select the most suitable source; the road who's traffic
        needs to be freed is selected.
        """
        
        # print('fitness: ', self.fitness)

        for source in self.fitness:
            source_fitness = self.fitness[source]

            total_fitness = sum(list(self.fitness.values()))
            probability = -1

            if not self.special_permission_vehicle_present:
                if source_fitness == 0:
                    probability = 0
                else:
                    source_fitness = 1 - source_fitness
                
                total_fitness = 1

            
            else:
                probability = source_fitness/total_fitness
            self.probabilities[source] = abs(probability)


    def memorize_best_source(self):
        """
        This method selects the food source with the highest calculated probability and
        returns it.
        """

        # print('probabilities: ', self.probabilities)
        max_probability_road = None
        max_probability = -1
        for source in self.probabilities:
            if self.probabilities[source] > max_probability:
                max_probability = self.probabilities[source]
                max_probability_road = source
            elif self.probabilities[source] == max_probability:
                if self.sources[source]['cars'] > self.sources[max_probability_road]['cars']:
                    max_probability = self.probabilities[source]
                    max_probability_road = source

        return max_probability_road


def tests():
    sources = {'road_id_1': {'cars': 11, 'special_permission_vehicles': 0, 'pedestrians': 0}, 
    'road_id_2': {'cars': 10, 'special_permission_vehicles': 0, 'pedestrians': 0}, 
    'road_id_3': {'cars': 100, 'special_permission_vehicles': 0, 'pedestrians': 0}}
    colony = abcModified(sources)
    colony.calculate_fitness()
    colony.calculate_probabilities()
    best_source = colony.memorize_best_source()

    print('\nsources: ', sources)
    print('fitness: ', colony.fitness)
    print('probabilities: ', colony.probabilities)
    print('best_source: ', best_source)

if __name__ == "__main__":
    tests()