data = open("zoo.data", "r")

all_boolean_data = open("zoo_boolean.txt", "w")

last_elem_data = open("last_elem.txt", "w")

name_data = open("names.txt", "w")

data_list = []


for d in data:
    #strip away the newline at the end so it does not end up in the list
    d = d.strip()
    #make list of all attributes
    d_list = d.split(",")
    #the boolean criteria now becomes true if we have more than 4 legs, otherwise false
    d_list[13] = "0" if int(d_list[13]) <= 4 else "1"
    #remove the last element to use for visualization later
    last_elem = d_list.pop()
    last_elem_data.write(last_elem)
    last_elem_data.write("\n")

    #remove first element which is the name of the point
    name = d_list.pop(0)
    name_data.write(name)
    name_data.write("\n")

    data_list.append(d_list)

for d in data_list:
    #turn the list back into a comma separated string
    d_string = ','.join(d)
    #write to separate data file
    all_boolean_data.write(d_string)
    all_boolean_data.write("\n")




#f.write("Woops! I have deleted the content!")
#f.close()