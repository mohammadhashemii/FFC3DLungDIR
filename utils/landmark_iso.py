import math


dirlab_info = {
    'case1': {'Spacing': [0.97, 0.97, 2.5]},
    'case2': {'Spacing': [1.16, 1.16, 2.5]},
    'case3': {'Spacing': [1.15, 1.15, 2.5]},
    'case4': {'Spacing': [1.13, 1.13, 2.5]},
    'case5': {'Spacing': [1.10, 1.10, 2.5]},
    'case6': {'Spacing': [0.97, 0.97, 2.5]},
    'case7': {'Spacing': [0.97, 0.97, 2.5]},
    'case8': {'Spacing': [0.97, 0.97, 2.5]},
    'case9': {'Spacing': [0.97, 0.97, 2.5]},
    'case10': {'Spacing': [0.97, 0.97, 2.5]}
}

def landmark_iso(input_file_path, input_resolution, output_file_path):

    # Read data from the input file
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Process the data
    modified_data = []
    for line in lines:
        # Split each line into a list of numbers
        numbers = list(map(int, line.split()))

        # Multiply each column by the specified factor
        modified_numbers = [numbers[0] * input_resolution[0],
                            numbers[1] * input_resolution[1],
                            numbers[2] * input_resolution[2]]

        # Append the modified numbers to the result list
        modified_data.append(modified_numbers)

    # Save the modified data to a new file
    with open(output_file_path, 'w') as file:
        for numbers in modified_data:
            # Convert the numbers to strings and join them with tabs
            line = '\t'.join(map(str, numbers)) + '\n'
            file.write(line)

    print("Data has been processed and saved to", output_file_path)


for i in range(1, 11):
    for p in ['00', '50']:
        old_landmarks_path = "./data/DIRLAB/Case" + str(i) + "Pack/ExtremePhases/Case" + str(i) + "_300_T" + p + "_xyz.txt"
        modified_landmarks_path = "./data/DIRLAB/points/case" + str(i) + "/case" + str(i) + "_300_T" + p + "_xyz_R.txt"
        landmark_iso(old_landmarks_path, dirlab_info['case'+ str(i)]['Spacing'], modified_landmarks_path)