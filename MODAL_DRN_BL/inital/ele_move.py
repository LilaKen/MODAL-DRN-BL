import re

# Clean impurity information from files
for i in range(1, 103059):
    # Read the content of the element file
    with open(f"../dataset/element/element{i}.txt", 'r') as e:
        pre_ele = e.read()

        # Extract a specific part of the string (if needed for debugging or other purposes)
        move_ele = pre_ele[313:334]

        # Remove the strings 'AREA' and 'ISTR' from the content
        content = pre_ele.replace('AREA', '')
        content_cleaned = content.replace('ISTR', '')

    # Write the cleaned content back to the file
    with open(f"../dataset/element/element{i}.txt", "w") as d:
        d.write(content_cleaned)
