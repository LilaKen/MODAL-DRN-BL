import numpy as np

# Initialize an array to hold the results
total_result = np.zeros((103058, 10))

# Loop through each sample file
for sample in range(1, 103059):
    # Open and read the result file
    with open(f"../dataset/result/result{sample}.txt", 'r') as r:
        res_result = r.read()
        # Extract relevant data
        result_pre_data = res_result[103:]
        result_medi_data = result_pre_data.split()

        # Convert data to float and store in an array
        add_result = np.array([float(result) for result in result_medi_data])

        # Extract every 5th value starting from the 2nd element
        empty_array = add_result[1::5]

        # Store the extracted values in the total_result array
        total_result[sample - 1, :] = empty_array

# Save the total_result array to a text file
np.savetxt("../dataset/meta.txt", total_result)
