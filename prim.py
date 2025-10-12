def calculate_distance(path, distance_matrix):
    total = 0
    for i in range(len(path) - 1):
        total += distance_matrix[path[i]][path[i + 1]]
    total += distance_matrix[path[-1]][path[0]]  # close the tour
    return total


def prim_mst(distance_matrix, start=0):
    n = len(distance_matrix)
    in_mst = [False] * n
    key = [float("inf")] * n
    parent = [-1] * n

    key[start] = 0
    for _ in range(n):
        u = -1
        best = float("inf")
        for v in range(n):
            if not in_mst[v] and key[v] < best:
                best, u = key[v], v
        in_mst[u] = True

        for v in range(n):
            w = distance_matrix[u][v]
            if not in_mst[v] and 0 <= w < key[v] and v != u:
                key[v] = w
                parent[v] = u
    return parent


def preorder_from_parent(parent, root=0):
    n = len(parent)
    adj = [[] for _ in range(n)]
    for v in range(n):
        p = parent[v]
        if p != -1:
            adj[p].append(v)
            adj[v].append(p)

    order, stack, seen = [], [root], [False] * n
    while stack:
        u = stack.pop()
        if seen[u]:
            continue
        seen[u] = True
        order.append(u)
        for w in sorted(adj[u], reverse=True):
            if not seen[w]:
                stack.append(w)
    return order


def solve_tsp_prim(distance_matrix, start=0):
    parent = prim_mst(distance_matrix, start)
    preorder = preorder_from_parent(parent, start)
    if preorder[0] != start:
        i = preorder.index(start)
        preorder = preorder[i:] + preorder[:i]

    path = preorder
    dist = calculate_distance(path, distance_matrix)
    return path, dist
