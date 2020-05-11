import os
import VehicleDetection as vd
from ABC import abcModified as abc

def main():

    # Current working directory
    CWD_PATH = os.path.dirname(os.path.realpath(__file__))

    # images for testing
        ## the static (hardcoded) images here could be replaced with anything else, possibly automated to supply images for vehicle detection
    test_images_path = CWD_PATH + "\\test_images"
        ## The first image would be regarded as road1, second as road2, and third as road3
    images = [test_images_path+"\\test1.jpg", test_images_path+"\\test2.jpg", test_images_path+"\\test3.jpg"]
    print(images)

    road_parameter_dict = {}
    i = 0
    for image in images:
        i += 1
        try:
            road_parameter_dict['road'+str(i)] = vd.compile_parameters(image)
        except Exception as e:
            print('error while compiling parameters from: ' + image)
            print(e)
    
    print(road_parameter_dict)
    abc_mod = abc(road_parameter_dict)
    abc_mod.calculate_fitness()
    abc_mod.calculate_probabilities()
    abc_decision = abc_mod.memorize_best_source()

    print(abc_decision)
    return abc_decision


if __name__ == "__main__":
    main()