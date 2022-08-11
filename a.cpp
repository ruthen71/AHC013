#include <bits/stdc++.h>
using namespace std;

#ifdef _RUTHEN
#include "../debug.hpp"
#else
#define show(...) true
#endif

#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx2")

using ll = long long;
#define rep(i, n) for (int i = 0; i < (n); i++)
template <class T> using V = vector<T>;

#include <atcoder/dsu>

#ifdef _RUTHEN
double time_limit = 60;
#else
double time_limit = 2.8;
#endif

// https://atcoder.jp/contests/ahc011/submissions/32267675
// Timer
inline ll GetTSC() {
    ll lo, hi;
    asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return lo + (hi << 32);
}
inline double GetSeconds() { return GetTSC() / 2.8e9; }

double saveTime;
void Init() { saveTime = GetSeconds(); }
double Elapsed() { return GetSeconds() - saveTime; }

// Random
const int MAX_RAND = 1 << 30;
struct Rand {
    ll x, y, z, w, o;
    Rand() {}
    Rand(ll seed) {
        reseed(seed);
        o = 0;
    }
    inline void reseed(ll seed) {
        x = 0x498b3bc5 ^ seed;
        y = 0;
        z = 0;
        w = 0;
        rep(i, 20) mix();
    }
    inline void mix() {
        ll t = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        w = w ^ (w >> 19) ^ t ^ (t >> 8);
    }
    inline ll rand() {
        mix();
        return x & (MAX_RAND - 1);
    }
    inline int nextInt(int n) { return rand() % n; }
    inline int nextInt(int L, int R) { return rand() % (R - L + 1) + L; }
    inline int nextBool() {
        if (o < 4) o = rand();
        o >>= 2;
        return o & 1;
    }
    double nextDouble() { return rand() * 1.0 / MAX_RAND; }
};
Rand my(2022);

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

int calc_score_fast(int N, vector<string> field, const Result &res, const int K) {
    for (auto r : res.move) {
        assert(field[r.before_row][r.before_col] != '0');
        assert(field[r.after_row][r.after_col] == '0');
        swap(field[r.before_row][r.before_col], field[r.after_row][r.after_col]);
    }

    vector<int> srvcnt(K, 0);
    vector<vector<int>> srvloc(N, vector<int>(N));
    rep(i, N) {
        rep(j, N) {
            if (field[i][j] != '0') {
                int srvtype = field[i][j] - '1';
                srvloc[i][j] = srvtype * 100 + srvcnt[srvtype];
                srvcnt[srvtype]++;
            }
        }
    }
    atcoder::dsu uf(100 * K);
    for (auto r : res.connect) {
        uf.merge(srvloc[r.c1_row][r.c1_col], srvloc[r.c2_row][r.c2_col]);
    }
    int score = 0;
    rep(i, 100 * K) {
        if (uf.leader(i) == i) score += uf.size(i) * (uf.size(i) - 1) / 2;
    }
    return max(score, 0);
}

void print_answer(const Result &res, const int K) {
    assert(res.move.size() + res.connect.size() <= 100 * K);
    cout << res.move.size() << '\n';
    for (auto m : res.move) {
        cout << m.before_row << " " << m.before_col << " " << m.after_row << " " << m.after_col << '\n';
    }
    cout << res.connect.size() << '\n';
    for (auto m : res.connect) {
        cout << m.c1_row << " " << m.c1_col << " " << m.c2_row << " " << m.c2_col << '\n';
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

    vector<MoveAction> modify(vector<MoveAction> &pre_moves) {
        int move_limit = (int)pre_moves.size();
        assert(move_limit <= 100 * K);
        int change = engine() % move_limit;
        vector<MoveAction> ret;
        double prob = my.nextDouble();  // 0.5 以上ならmove 数を減らす 0.5未満ならmove数を増やす
        for (int i = 0; i < move_limit; i++) {
            if (i == change) {
                if (prob > 0.5 and move_limit > 1) continue;
                // change
                // 高速化の余地がある
                while (true) {
                    int row = engine() % N;
                    int col = engine() % N;
                    int dir = engine() % 4;
                    if (field[row][col] != '0' && can_move(row, col, dir)) {
                        swap(field[row][col], field[row + DR[dir]][col + DC[dir]]);
                        ret.emplace_back(row, col, row + DR[dir], col + DC[dir]);
                        break;
                    }
                }
            } else {
                auto [row, col, nrow, ncol] = pre_moves[i];
                int dir = -1;
                rep(k, 4) {
                    if (row + DR[k] == nrow and col + DC[k] == ncol) {
                        dir = k;
                        break;
                    }
                }
                if (field[row][col] != '0' && can_move(row, col, dir)) {
                    swap(field[row][col], field[row + DR[dir]][col + DC[dir]]);
                    ret.emplace_back(row, col, row + DR[dir], col + DC[dir]);
                } else {
                    while (true) {
                        int row = engine() % N;
                        int col = engine() % N;
                        int dir = engine() % 4;
                        if (field[row][col] != '0' && can_move(row, col, dir)) {
                            swap(field[row][col], field[row + DR[dir]][col + DC[dir]]);
                            ret.emplace_back(row, col, row + DR[dir], col + DC[dir]);
                            break;
                        }
                    }
                }
            }
        }
        if (prob <= 0.5 and (int) ret.size() < 100 * K) {
            while (true) {
                int row = engine() % N;
                int col = engine() % N;
                int dir = engine() % 4;
                if (field[row][col] != '0' && can_move(row, col, dir)) {
                    swap(field[row][col], field[row + DR[dir]][col + DC[dir]]);
                    ret.emplace_back(row, col, row + DR[dir], col + DC[dir]);
                    break;
                }
            }
        }
        assert(abs((int)ret.size() - move_limit) <= 1);
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
        int iter_count = 0;
        while (true) {
            iter_count++;
            if (Elapsed() <= time_limit) {
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
                print_answer(res, K);
#endif
            } else {
                break;
            }
        }
        cerr << "iter_count = " << iter_count << '\n';
        return max_res;
    }

    Result solve_mountain() {
        int max_score = 0;
        Result max_res;
        // initialize
        {
            auto moves = move();
            auto connects = connect((int)moves.size());
            max_res = Result(moves, connects);
            field = field_backup;
            max_score = calc_score_fast(N, field, max_res, K);
            field = field_backup;
        }
        int iter_count = 1;
        while (true) {
            iter_count++;
            if (Elapsed() <= time_limit) {
                // modify move
                auto moves = modify(max_res.move);
                // from each computer, connect to right and/or bottom if it will reach the same type
                auto connects = connect((int)moves.size());
                Result res = Result(moves, connects);
                field = field_backup;
                // int score = calc_score(N, field, res);
                int score = calc_score_fast(N, field, res, K);
                if (score > max_score) {
                    // cerr << "iter_count = " << iter_count << '\n';
                    max_score = score;
                    max_res = res;
                }
#ifdef _RUTHEN
                // print_answer(res);
#endif
            } else {
                break;
            }
        }
        cerr << "iter_count = " << iter_count << '\n';
        return max_res;
    }

    Result solve_sa() {
        // initialize
        int max_score = 0;
        Result max_res;
        {
            auto moves = move();
            auto connects = connect((int)moves.size());
            max_res = Result(moves, connects);
            field = field_backup;
            max_score = calc_score_fast(N, field, max_res, K);
            field = field_backup;
        }
        double start_temp = 1, end_temp = 0;
        int iter_count = 1;
        while (true) {
            iter_count++;
            double now_time = Elapsed();
            if (now_time > time_limit) break;
#if 0
            double last_time = now_time;
            double next_time;
#endif
            // modify move
            auto moves = modify(max_res.move);
#if 0
            if (iter_count % 1000 == 0) {
                next_time = Elapsed();
                show(next_time - last_time);
                last_time = next_time;
            }
#endif
            // from each computer, connect to right and/or bottom if it will reach the same type
            auto connects = connect((int)moves.size());
#if 0
            if (iter_count % 1000 == 0) {
                next_time = Elapsed();
                show(next_time - last_time);
                last_time = next_time;
            }
#endif
            Result res = Result(moves, connects);
            field = field_backup;
            // int score = calc_score(N, field, res);
            int score = calc_score_fast(N, field, res, K);
#if 0
            if (iter_count % 1000 == 0) {
                next_time = Elapsed();
                show(next_time - last_time);
                last_time = next_time;
            }
#endif
            // 温度関数
            double temp = start_temp + (end_temp - start_temp) * now_time / time_limit;
            // 遷移確率関数
            double prob = exp((score - max_score) / temp);
            if (prob > my.nextDouble()) {
                // cerr << "iter_count = " << iter_count << '\n';
                max_score = score;
                max_res = res;
#ifdef _RUTHEN
                // print_answer(res, K);
#endif
            }
        }
        cerr << "iter_count = " << iter_count << '\n';
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
    // auto ret = s.solve_random();
    // auto ret = s.solve_mountain();
    auto ret = s.solve_sa();

    cerr << "Score = " << calc_score_fast(N, field, ret, K) << '\n';

    print_answer(ret, K);
    return 0;
}
