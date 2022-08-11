#include <bits/stdc++.h>
using namespace std;

#ifdef _RUTHEN
#include "../debug.hpp"
#else
#define show(...) true
#endif

using ll = long long;
#define rep(i, n) for (int i = 0; i < (n); i++)
template <class T> using V = vector<T>;

// https://atcoder.jp/contests/ahc011/submissions/32267675
inline ll GetTSC() {
    ll lo, hi;
    asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return lo + (hi << 32);
}
inline double GetSeconds() { return GetTSC() / 2.8e9; }

double saveTime;
void Init() { saveTime = GetSeconds(); }
double Elapsed() { return GetSeconds() - saveTime; }

struct MoveAction {
    int before_row, before_col, after_row, after_col;
    MoveAction(int before_row, int before_col, int after_row, int after_col) : before_row(before_row), before_col(before_col), after_row(after_row), after_col(after_col) {}
};

struct ConnectAction {
    int c1_row, c1_col, c2_row, c2_col;
    ConnectAction(int c1_row, int c1_col, int c2_row, int c2_col) : c1_row(c1_row), c1_col(c1_col), c2_row(c2_row), c2_col(c2_col) {}
};

struct Result {
    vector<MoveAction> move;
    vector<ConnectAction> connect;
    Result() {}
    Result(const vector<MoveAction> &move, const vector<ConnectAction> &con) : move(move), connect(con) {}
};

struct UnionFind {
    map<pair<int, int>, pair<int, int>> parent;
    UnionFind() : parent() {}

    pair<int, int> find(pair<int, int> x) {
        if (parent.find(x) == parent.end()) {
            parent[x] = x;
            return x;
        } else if (parent[x] == x) {
            return x;
        } else {
            parent[x] = find(parent[x]);
            return parent[x];
        }
    }

    void unite(pair<int, int> x, pair<int, int> y) {
        x = find(x);
        y = find(y);
        if (x != y) {
            parent[x] = y;
        }
    }
};

int calc_score(int N, vector<string> field, const Result &res) {
    for (auto r : res.move) {
        assert(field[r.before_row][r.before_col] != '0');
        assert(field[r.after_row][r.after_col] == '0');
        swap(field[r.before_row][r.before_col], field[r.after_row][r.after_col]);
    }

    UnionFind uf;
    for (auto r : res.connect) {
        pair<int, int> p1(r.c1_row, r.c1_col), p2(r.c2_row, r.c2_col);
        uf.unite(p1, p2);
    }

    vector<pair<int, int>> computers;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (field[i][j] != '0') {
                computers.emplace_back(i, j);
            }
        }
    }

    int score = 0;
    for (int i = 0; i < (int)computers.size(); i++) {
        for (int j = i + 1; j < (int)computers.size(); j++) {
            auto c1 = computers[i];
            auto c2 = computers[j];
            if (uf.find(c1) == uf.find(c2)) {
                score += (field[c1.first][c1.second] == field[c2.first][c2.second]) ? 1 : -1;
            }
        }
    }

    return max(score, 0);
}

void print_answer(const Result &res) {
    cout << res.move.size() << endl;
    for (auto m : res.move) {
        cout << m.before_row << " " << m.before_col << " " << m.after_row << " " << m.after_col << endl;
    }
    cout << res.connect.size() << endl;
    for (auto m : res.connect) {
        cout << m.c1_row << " " << m.c1_col << " " << m.c2_row << " " << m.c2_col << endl;
    }
}

struct Solver {
    static constexpr char USED = 'x';
    static constexpr int DR[4] = {0, 1, 0, -1};
    static constexpr int DC[4] = {1, 0, -1, 0};

    int N, K;
    int action_count_limit;
    mt19937 engine;
    vector<string> field;
    vector<string> field_backup;

    Solver(int N, int K, const vector<string> &field, int seed = 0) : N(N), K(K), action_count_limit(K * 100), field(field) {
        engine.seed(seed);
        field_backup = field;
    }

    bool can_move(int row, int col, int dir) const {
        int nrow = row + DR[dir];
        int ncol = col + DC[dir];
        if (0 <= nrow && nrow < N && 0 <= ncol && ncol < N) {
            return field[nrow][ncol] == '0';
        }
        return false;
    }

    vector<MoveAction> move(int move_limit = -1) {
        vector<MoveAction> ret;
        if (move_limit == -1) {
            move_limit = K * 50;
        }
        assert(action_count_limit >= move_limit);
        for (int i = 0; i < move_limit; i++) {
            int row = engine() % N;
            int col = engine() % N;
            int dir = engine() % 4;
            if (field[row][col] != '0' && can_move(row, col, dir)) {
                swap(field[row][col], field[row + DR[dir]][col + DC[dir]]);
                ret.emplace_back(row, col, row + DR[dir], col + DC[dir]);
            }
        }
        return ret;
    }

    bool can_connect(int row, int col, int dir) const {
        int nrow = row + DR[dir];
        int ncol = col + DC[dir];
        while (0 <= nrow && nrow < N && 0 <= ncol && ncol < N) {
            if (field[nrow][ncol] == field[row][col]) {
                return true;
            } else if (field[nrow][ncol] != '0') {
                return false;
            }
            nrow += DR[dir];
            ncol += DC[dir];
        }
        return false;
    }

    ConnectAction line_fill(int row, int col, int dir) {
        int nrow = row + DR[dir];
        int ncol = col + DC[dir];
        while (0 <= nrow && nrow < N && 0 <= ncol && ncol < N) {
            if (field[nrow][ncol] == field[row][col]) {
                return ConnectAction(row, col, nrow, ncol);
            }
            assert(field[nrow][ncol] == '0');
            field[nrow][ncol] = USED;
            nrow += DR[dir];
            ncol += DC[dir];
        }
        assert(false);
    }

    vector<ConnectAction> connect(int move_count) {
        int connect_count_limit = action_count_limit - move_count;
        vector<ConnectAction> ret;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (field[i][j] != '0' && field[i][j] != 'x') {
                    for (int dir = 0; dir < 2; dir++) {
                        if (can_connect(i, j, dir)) {
                            ret.push_back(line_fill(i, j, dir));
                            connect_count_limit--;
                            if (connect_count_limit <= 0) {
                                return ret;
                            }
                        }
                    }
                }
            }
        }
        return ret;
    }

    Result solve_random() {
        int max_score = 0;
        Result max_res;
        while (true) {
            if (Elapsed() <= 2.8) {
                // create random moves
                auto moves = move();
                // from each computer, connect to right and/or bottom if it will reach the same type
                auto connects = connect((int)moves.size());
                Result res = Result(moves, connects);
                field = field_backup;
                int score = calc_score(N, field, res);
                if (score > max_score) {
                    max_score = score;
                    max_res = res;
                }
#ifdef _RUTHEN
                print_answer(res);
#endif
            } else {
                break;
            }
        }
        return max_res;
    }
};

int main() {
    Init();
    int N, K;
    cin >> N >> K;
    vector<string> field(N);
    for (int i = 0; i < N; i++) {
        cin >> field[i];
    }

    Solver s(N, K, field);
    auto ret = s.solve_random();
    // auto ret = s.solve_mountain();

    cerr << "Score = " << calc_score(N, field, ret) << endl;

    print_answer(ret);

    return 0;
}
