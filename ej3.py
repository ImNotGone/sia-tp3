
def load_numbers(file_name, rows):
    arr = []

    # read from file mats of size rows x N and return them in an array
    with open(file_name, "r") as data:
        mat = []
        i = 0
        for data_row in data:
            row = []

            # remove spaces and \n
            pruned_row = data_row.replace(" ", "").replace('\n', "")
            for data_col in pruned_row:
                row += [int(data_col)]

            mat += [row]
            i += 1
            # if i loaded 'rows' rows -> load the matrix to the arr
            if(i % rows == 0):
                arr += [mat]
                mat = []

        # load what remains
        arr += [mat]
    return arr

file_name = "./data/ej3-digitos.txt"
rows = 7
numbers = load_numbers(file_name, rows)

