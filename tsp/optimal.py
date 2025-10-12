# Original TSPLIB problems found at https://tsplib95.readthedocs.io/en/stable/pages/usage.html#usage
# Returns: Optimal solutions path from corresponding TSPLIB adjacency matrix
# Based on original files: *.opt.tour, index adjusted to start at 0 

bays29opt = [x-1 for x in [
    1, 28, 6, 12, 9, 5, 26, 29, 3, 2, 20, 10, 4, 15, 18,
    17, 14, 22, 11, 19, 25, 7, 23, 27, 8, 24, 16, 13, 21
]]

berlin52opt = [x-1 for x in [
    1, 49, 32, 45, 19, 41, 8, 9,
    10, 43, 33, 51, 11, 52, 14, 13,
    47, 26, 27, 28, 12, 25, 4, 6,
    15, 5, 24, 48, 38, 37, 40, 39,
    36, 35, 34, 44, 46, 16, 29, 50,
    20, 23, 30, 2, 7, 42, 21, 17,
    3, 18, 31, 22 
]]

eil76opt = [x-1 for x in [
    1, 33, 63, 16, 3, 44, 32, 9, 39, 72, 58,
    10, 31, 55, 25, 50, 18, 24, 49, 23, 56, 41,
    43, 42, 64, 22, 61, 21, 47, 36, 69, 71, 60,
    70, 20, 37, 5, 15, 57, 13, 54, 19, 14, 59,
    66, 65, 38, 11, 53, 7, 35, 8, 46, 34, 52,
    27, 45, 29, 48, 30, 4, 75, 76, 67, 26, 12,
    40, 17, 51, 6, 68, 2, 74, 28, 62, 73
]]

ulysses16opt = [x-1 for x in [
    1, 14, 13, 12, 7, 6, 15, 5, 
    11, 9, 10, 16, 3, 2, 4, 8
]]

