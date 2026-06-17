#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>

using namespace std;

const int INF = numeric_limits<int>::max();

struct Edge {
    int to;
    int weight;
};

vector<int> dijkstra(int start, const vector<vector<Edge>>& graph) {
    int n = graph.size();
    vector<int> dist(n, INF);
    dist[start] = 0;
    // priority_queue: (distance, vertex), min-heap
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue; // устаревшая запись
        for (const auto& e : graph[u]) {
            if (dist[u] + e.weight < dist[e.to]) {
                dist[e.to] = dist[u] + e.weight;
                pq.push({dist[e.to], e.to});
            }
        }
    }
    return dist;
}

int main() {
    // Пример: 5 вершин, ориентированный взвешенный граф
    int n = 5;
    vector<vector<Edge>> graph(n);
    graph[0].push_back({1, 10});
    graph[0].push_back({2, 3});
    graph[1].push_back({3, 2});
    graph[2].push_back({1, 4});
    graph[2].push_back({3, 8});
    graph[2].push_back({4, 2});
    graph[3].push_back({4, 5});

    auto dist = dijkstra(0, graph);

    cout << "Кратчайшие расстояния от вершины 0:\n";
    for (int i = 0; i < n; ++i) {
        cout << "  до " << i << ": " << (dist[i] == INF ? "∞" : to_string(dist[i])) << "\n";
    }
    return 0;
}
