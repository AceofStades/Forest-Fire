import heapq
import numpy as np

class DStarLite:
    def __init__(self, grid, start, goal):
        self.grid = grid  # 320x400 probability grid
        self.start = start
        self.goal = goal
        self.km = 0
        self.g = {}   # Actual cost from goal to node
        self.rhs = {} # One-step lookahead cost
        self.queue = []
        
        # Initialize the entire grid
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                self.g[(i, j)] = float('inf')
                self.rhs[(i, j)] = float('inf')
        
        # Goal is the center of search (Reverse Search)
        self.rhs[self.goal] = 0
        self.push_to_queue(self.goal, self.calculate_key(self.goal))

    def calculate_key(self, s):
        """Standard D* Lite key calculation."""
        h = np.sqrt((self.start[0] - s[0])**2 + (self.start[1] - s[1])**2)
        k1 = min(self.g[s], self.rhs[s]) + h + self.km
        k2 = min(self.g[s], self.rhs[s])
        return (k1, k2)

    def push_to_queue(self, s, key):
        heapq.heappush(self.queue, (key, s))

    def get_neighbors(self, u):
        """8-way connectivity for natural movement."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                x, y = u[0] + dx, u[1] + dy
                if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                    neighbors.append((x, y))
        return neighbors

    def update_vertex(self, u):
        """Maintains local consistency."""
        if u != self.goal:
            min_rhs = float('inf')
            for v in self.get_neighbors(u):
                # Edge cost = distance (1) + penalty based on fire prob
                prob = self.grid[v[0], v[1]]
                cost = 1 + (prob * 1000 if prob > 0.6 else prob * 10)
                min_rhs = min(min_rhs, cost + self.g[v])
            self.rhs[u] = min_rhs
        
        # Priority Queue update
        self.queue = [item for item in self.queue if item[1] != u]
        heapq.heapify(self.queue)
        if self.g[u] != self.rhs[u]:
            self.push_to_queue(u, self.calculate_key(u))

    def compute_shortest_path(self):
        """Expands nodes until start node is consistent."""
        while (len(self.queue) > 0 and 
               (self.queue[0][0] < self.calculate_key(self.start) or 
                self.rhs[self.start] != self.g[self.start])):
            k_old, u = heapq.heappop(self.queue)
            k_new = self.calculate_key(u)
            
            if k_old < k_new:
                self.push_to_queue(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for v in self.get_neighbors(u):
                    self.update_vertex(v)
            else:
                self.g[u] = float('inf')
                for v in [u] + self.get_neighbors(u):
                    self.update_vertex(v)