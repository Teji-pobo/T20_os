/*
  T20 Cricket Simulator
  
 Assumptions:
 T20 format so 20 overs per innings and 10 wickets per team 
 In case of a tie 6 overs for powerplay alloted 
 16-20 are death overs so death bowlers are prioritised
 Each bowler has a max of 4 overs and no consecutive overs 
 The pitch is the critical section protected by mutex
 Crease capacity is 2 enforced by semaphore
 Each team have exactly 5 bowlers 
 Fielders sleep until ball is in air using condition variable
 Run out deadlock detected using wait for graph with DFS
 User selects FCFS or SJF batting schedule per innings
 Bowler scheduling uses round robin then priority in death overs
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>

#define MAX_OVERS 20
#define BALLS_PER_OVER 6
#define MAX_WICKETS 10
#define NUM_FIELDERS 10
#define NUM_BOWLERS 5
#define NUM_BATSMEN 11
#define TIME_QUANTUM 6
#define CREASE_CAPACITY 2
#define DEATH_OVER_START 16
#define POWERPLAY_END 6
#define HIGH_INTENSITY_THRESHOLD 70

#define SCHED_FCFS 1
#define SCHED_SJF 2

#define OUTCOME_DOT 0
#define OUTCOME_SINGLE 1
#define OUTCOME_DOUBLE 2
#define OUTCOME_TRIPLE 3
#define OUTCOME_FOUR 4
#define OUTCOME_SIX 5
#define OUTCOME_WICKET 6
#define OUTCOME_WIDE 7
#define OUTCOME_NOBALL 8
#define OUTCOME_LEG_BYE 9

typedef struct {
    int id;
    char name[50];
    int runs_scored;
    int balls_faced;
    int fours;
    int sixes;
    bool is_out;
    int stay_duration;
    int priority;
    bool is_tail_ender;
    char dismissal[100];
    int original_order;
    float strike_rate;
} Batsman;

typedef struct {
    int id;
    char name[50];
    int overs_bowled;
    int balls_in_current_over;
    int runs_given;
    int wickets;
    int maidens;
    int dots;
    int wides;
    int noballs;
    bool is_death_specialist;
    bool is_powerplay_specialist;
    int priority;
    float economy;
} Bowler;

typedef struct {
    int id;
    char name[50];
    char position[30];
    bool is_wicketkeeper;
    int catches;
    int runouts;
    int stumpings;
    bool is_active;
} Fielder;

typedef struct {
    int striker_idx;
    int non_striker_idx;
    int bowler_idx;
    char striker_name[50];
    char non_striker_name[50];
    char bowler_name[50];
    int over_number;
    int ball_number;
    int score_before;
    int wickets_before;
} BallContext;

typedef struct {
    int ball_number;
    int ball_type;
    int ball_speed;
    int ball_line;
    int ball_length;
    bool is_bowled;
    bool is_played;
    bool ball_in_air;
    int runs_scored;
    bool is_wide;
    bool is_noball;
    bool is_wicket;
    bool is_leg_bye;
    int wicket_type;
    char description[300];
    char outcome_str[50];
    char commentary[200];
    char dismissed_batsman_name[50];
    int dismissed_batsman_idx;
    BallContext context;
} Ball;

typedef struct {
    int total_runs;
    int wickets;
    int current_over;
    int current_ball;
    int extras;
    int wides;
    int noballs;
    int leg_byes;
    int target;
    bool is_first_innings;
    int match_intensity;
    bool match_over;
    char team_batting[50];
    char team_bowling[50];
    int batting_sched_type;
    float current_run_rate;
    float required_run_rate;
    int balls_remaining;
    int runs_in_current_over;
} MatchState;

typedef struct {
    int adj_matrix[NUM_BATSMEN][NUM_BATSMEN];
    bool is_waiting[NUM_BATSMEN];
    int waiting_for_resource[NUM_BATSMEN];
    int holds_resource[NUM_BATSMEN];
} WaitForGraph;

typedef struct {
    int scheduling_algo;
    int time_quantum;
    int current_quantum_used;
    int bowler_queue[NUM_BOWLERS];
    int bowler_queue_front;
    int bowler_queue_rear;
    int last_bowler;
} SchedulerState;

typedef struct {
    int occupant_id;
    bool is_locked;
    pthread_mutex_t lock;
} CreaseEnd;

typedef struct {
    char name[50];
    char short_name[10];
    Batsman batsmen[NUM_BATSMEN];
    Batsman original_batsmen[NUM_BATSMEN];
    Bowler bowlers[NUM_BOWLERS];
    Fielder fielders[NUM_FIELDERS];
    int innings_score;
    int innings_wickets;
    int innings_overs;
    int innings_balls;
    int innings_extras;
    float innings_run_rate;
} Team;

Ball pitch_ball;
MatchState match_state;
SchedulerState scheduler;
WaitForGraph wait_graph;
CreaseEnd crease_ends[2];

Team teams[2];
int batting_team_idx = 0;
int bowling_team_idx = 1;

Batsman* batsmen;
Bowler* bowlers;
Fielder* fielders;

int striker_idx = 0;
int non_striker_idx = 1;
int current_bowler_idx = 0;
int next_batsman_idx = 2;

pthread_mutex_t pitch_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t score_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t match_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t ball_state_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t scheduler_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t waitgraph_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t index_mutex = PTHREAD_MUTEX_INITIALIZER;

sem_t crease_semaphore;
sem_t ball_ready_sem;
sem_t stroke_complete_sem;
sem_t over_complete_sem;

pthread_cond_t ball_hit_cond = PTHREAD_COND_INITIALIZER;
pthread_cond_t ball_bowled_cond = PTHREAD_COND_INITIALIZER;
pthread_cond_t fielder_wake_cond = PTHREAD_COND_INITIALIZER;

volatile bool match_running = true;
volatile bool ball_in_air = false;
volatile bool innings_over = false;

int total_balls_bowled = 0;
int total_boundaries = 0;
int total_sixes = 0;

void initialize_teams(void);
void initialize_match_state(void);
void initialize_scheduler(void);
void initialize_synchronization(void);
void initialize_wait_graph(void);
void reset_for_innings(int bat_team, int bowl_team);
void apply_batting_schedule(int team_idx, int sched_type);
void apply_fcfs_order(int team_idx);
void apply_sjf_order(int team_idx);

void capture_ball_context(BallContext* ctx);
void context_switch_bowler(void);
int select_next_bowler_rr(void);
int select_next_bowler_priority(void);

int generate_ball_outcome(const BallContext* ctx);
void process_ball_outcome(int outcome, const BallContext* ctx);
void switch_strike(void);
void new_batsman(void);
void calculate_match_intensity(void);
void update_run_rates(void);

void* bowler_thread_func(void* arg);
void* batsman_thread_func(void* arg);
void* fielder_thread_func(void* arg);
void* umpire_thread_func(void* arg);

void add_edge_waitgraph(int from, int to);
void remove_edge_waitgraph(int from, int to);
bool detect_cycle_dfs(int node, bool visited[], bool rec_stack[]);
bool detect_deadlock(void);
int resolve_deadlock(void);
void handle_runout_scenario(const BallContext* ctx);

void print_ball_log(const BallContext* ctx);
void print_over_summary(void);
void print_scorecard(void);
void print_final_result(void);
void cleanup_resources(void);

void simulate_innings(int innings_num);
int get_scheduling_choice(int innings_num);

void initialize_teams(void) {
    strcpy(teams[0].name, "INDIA");
    strcpy(teams[0].short_name, "IND");
    
    const char* india_batsmen[] = {
        "Rohit Sharma", "Yashasvi Jaiswal", "Virat Kohli", "Suryakumar Yadav",
        "Hardik Pandya", "Rishabh Pant", "Ravindra Jadeja", "Axar Patel",
        "Kuldeep Yadav", "Jasprit Bumrah", "Mohammed Siraj"
    };
    const int india_stay[] = {35, 30, 40, 28, 25, 22, 18, 15, 8, 5, 4};
    
    const char* india_bowlers[] = {
        "Jasprit Bumrah", "Mohammed Siraj", "Arshdeep Singh",
        "Kuldeep Yadav", "Ravindra Jadeja"
    };
    
    strcpy(teams[1].name, "AUSTRALIA");
    strcpy(teams[1].short_name, "AUS");
    
    const char* aus_batsmen[] = {
        "Travis Head", "David Warner", "Mitchell Marsh", "Steve Smith",
        "Glenn Maxwell", "Marcus Stoinis", "Tim David", "Pat Cummins",
        "Mitchell Starc", "Adam Zampa", "Josh Hazlewood"
    };
    const int aus_stay[] = {32, 28, 35, 30, 20, 22, 18, 12, 6, 4, 3};
    
    const char* aus_bowlers[] = {
        "Pat Cummins", "Mitchell Starc", "Josh Hazlewood",
        "Adam Zampa", "Glenn Maxwell"
    };
    
    const char* fielder_positions[] = {
        "Slip", "Gully", "Point", "Cover", "Mid-off",
        "Mid-on", "Midwicket", "Square Leg", "Fine Leg", "Wicketkeeper"
    };
    
    const char** bat_names[2] = {india_batsmen, aus_batsmen};
    const char** bowl_names[2] = {india_bowlers, aus_bowlers};
    const int* stay_durations[2] = {india_stay, aus_stay};
    
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < NUM_BATSMEN; i++) {
            teams[t].batsmen[i].id = i;
            teams[t].batsmen[i].original_order = i;
            strncpy(teams[t].batsmen[i].name, bat_names[t][i], 49);
            teams[t].batsmen[i].name[49] = '\0';
            teams[t].batsmen[i].runs_scored = 0;
            teams[t].batsmen[i].balls_faced = 0;
            teams[t].batsmen[i].fours = 0;
            teams[t].batsmen[i].sixes = 0;
            teams[t].batsmen[i].is_out = false;
            teams[t].batsmen[i].stay_duration = stay_durations[t][i];
            teams[t].batsmen[i].priority = (i < 4) ? 1 : (i < 7) ? 2 : 3;
            teams[t].batsmen[i].is_tail_ender = (i >= 8);
            teams[t].batsmen[i].strike_rate = 0.0;
            strcpy(teams[t].batsmen[i].dismissal, "not out");
            teams[t].original_batsmen[i] = teams[t].batsmen[i];
        }
        
        for (int i = 0; i < NUM_BOWLERS; i++) {
            teams[t].bowlers[i].id = i;
            strncpy(teams[t].bowlers[i].name, bowl_names[t][i], 49);
            teams[t].bowlers[i].name[49] = '\0';
            teams[t].bowlers[i].overs_bowled = 0;
            teams[t].bowlers[i].balls_in_current_over = 0;
            teams[t].bowlers[i].runs_given = 0;
            teams[t].bowlers[i].wickets = 0;
            teams[t].bowlers[i].maidens = 0;
            teams[t].bowlers[i].dots = 0;
            teams[t].bowlers[i].wides = 0;
            teams[t].bowlers[i].noballs = 0;
            teams[t].bowlers[i].is_death_specialist = (i == 0 || i == 1);
            teams[t].bowlers[i].is_powerplay_specialist = (i == 1 || i == 2);
            teams[t].bowlers[i].priority = teams[t].bowlers[i].is_death_specialist ? 1 : 2;
            teams[t].bowlers[i].economy = 0.0;
        }
        
        for (int i = 0; i < NUM_FIELDERS; i++) {
            teams[t].fielders[i].id = i;
            snprintf(teams[t].fielders[i].name, 50, "Fielder %d", i + 1);
            strncpy(teams[t].fielders[i].position, fielder_positions[i], 29);
            teams[t].fielders[i].position[29] = '\0';
            teams[t].fielders[i].is_wicketkeeper = (i == 9);
            teams[t].fielders[i].catches = 0;
            teams[t].fielders[i].runouts = 0;
            teams[t].fielders[i].stumpings = 0;
            teams[t].fielders[i].is_active = false;
        }
        
        teams[t].innings_score = 0;
        teams[t].innings_wickets = 0;
        teams[t].innings_overs = 0;
        teams[t].innings_balls = 0;
        teams[t].innings_extras = 0;
        teams[t].innings_run_rate = 0.0;
    }
}

void initialize_match_state(void) {
    match_state.total_runs = 0;
    match_state.wickets = 0;
    match_state.current_over = 0;
    match_state.current_ball = 0;
    match_state.extras = 0;
    match_state.wides = 0;
    match_state.noballs = 0;
    match_state.leg_byes = 0;
    match_state.target = 0;
    match_state.is_first_innings = true;
    match_state.match_intensity = 0;
    match_state.match_over = false;
    match_state.batting_sched_type = SCHED_FCFS;
    match_state.current_run_rate = 0.0;
    match_state.required_run_rate = 0.0;
    match_state.balls_remaining = MAX_OVERS * BALLS_PER_OVER;
    match_state.runs_in_current_over = 0;
    
    memset(&pitch_ball, 0, sizeof(Ball));
    pitch_ball.dismissed_batsman_idx = -1;
    
    total_balls_bowled = 0;
    total_boundaries = 0;
    total_sixes = 0;
}

void initialize_scheduler(void) {
    scheduler.scheduling_algo = 0;
    scheduler.time_quantum = TIME_QUANTUM;
    scheduler.current_quantum_used = 0;
    scheduler.last_bowler = -1;
    
    for (int i = 0; i < NUM_BOWLERS; i++) {
        scheduler.bowler_queue[i] = i;
    }
    scheduler.bowler_queue_front = 0;
    scheduler.bowler_queue_rear = NUM_BOWLERS - 1;
}

void initialize_synchronization(void) {
    sem_init(&crease_semaphore, 0, CREASE_CAPACITY);
    sem_init(&ball_ready_sem, 0, 0);
    sem_init(&stroke_complete_sem, 0, 0);
    sem_init(&over_complete_sem, 0, 0);
    
    for (int i = 0; i < 2; i++) {
        crease_ends[i].occupant_id = -1;
        crease_ends[i].is_locked = false;
        pthread_mutex_init(&crease_ends[i].lock, NULL);
    }
}

void initialize_wait_graph(void) {
    for (int i = 0; i < NUM_BATSMEN; i++) {
        wait_graph.is_waiting[i] = false;
        wait_graph.waiting_for_resource[i] = -1;
        wait_graph.holds_resource[i] = -1;
        for (int j = 0; j < NUM_BATSMEN; j++) {
            wait_graph.adj_matrix[i][j] = 0;
        }
    }
}

void reset_for_innings(int bat_team, int bowl_team) {
    batting_team_idx = bat_team;
    bowling_team_idx = bowl_team;
    
    batsmen = teams[bat_team].batsmen;
    bowlers = teams[bowl_team].bowlers;
    fielders = teams[bowl_team].fielders;
    
    for (int i = 0; i < NUM_BATSMEN; i++) {
        batsmen[i].runs_scored = 0;
        batsmen[i].balls_faced = 0;
        batsmen[i].fours = 0;
        batsmen[i].sixes = 0;
        batsmen[i].is_out = false;
        batsmen[i].strike_rate = 0.0;
        strcpy(batsmen[i].dismissal, "not out");
    }
    
    for (int i = 0; i < NUM_BOWLERS; i++) {
        bowlers[i].overs_bowled = 0;
        bowlers[i].balls_in_current_over = 0;
        bowlers[i].runs_given = 0;
        bowlers[i].wickets = 0;
        bowlers[i].maidens = 0;
        bowlers[i].dots = 0;
        bowlers[i].wides = 0;
        bowlers[i].noballs = 0;
        bowlers[i].economy = 0.0;
    }
    
    striker_idx = 0;
    non_striker_idx = 1;
    current_bowler_idx = 0;
    next_batsman_idx = 2;
    
    match_state.total_runs = 0;
    match_state.wickets = 0;
    match_state.current_over = 0;
    match_state.current_ball = 0;
    match_state.extras = 0;
    match_state.wides = 0;
    match_state.noballs = 0;
    match_state.leg_byes = 0;
    match_state.match_intensity = 0;
    match_state.current_run_rate = 0.0;
    match_state.balls_remaining = MAX_OVERS * BALLS_PER_OVER;
    match_state.runs_in_current_over = 0;
    
    strcpy(match_state.team_batting, teams[bat_team].name);
    strcpy(match_state.team_bowling, teams[bowl_team].name);
    
    scheduler.current_quantum_used = 0;
    scheduler.bowler_queue_front = 0;
    scheduler.last_bowler = -1;
    
    crease_ends[0].occupant_id = striker_idx;
    crease_ends[1].occupant_id = non_striker_idx;
    
    initialize_wait_graph();
    wait_graph.holds_resource[striker_idx] = 0;
    wait_graph.holds_resource[non_striker_idx] = 1;
    
    int val;
    sem_getvalue(&ball_ready_sem, &val);
    while (val > 0) { sem_wait(&ball_ready_sem); sem_getvalue(&ball_ready_sem, &val); }
    sem_getvalue(&stroke_complete_sem, &val);
    while (val > 0) { sem_wait(&stroke_complete_sem); sem_getvalue(&stroke_complete_sem, &val); }
    
    memset(&pitch_ball, 0, sizeof(Ball));
    pitch_ball.dismissed_batsman_idx = -1;
    
    match_running = true;
    innings_over = false;
    ball_in_air = false;
    
    total_balls_bowled = 0;
}

void apply_fcfs_order(int team_idx) {
    for (int i = 0; i < NUM_BATSMEN; i++) {
        teams[team_idx].batsmen[i] = teams[team_idx].original_batsmen[i];
        teams[team_idx].batsmen[i].runs_scored = 0;
        teams[team_idx].batsmen[i].balls_faced = 0;
        teams[team_idx].batsmen[i].fours = 0;
        teams[team_idx].batsmen[i].sixes = 0;
        teams[team_idx].batsmen[i].is_out = false;
        teams[team_idx].batsmen[i].strike_rate = 0.0;
        strcpy(teams[team_idx].batsmen[i].dismissal, "not out");
    }
}

void apply_sjf_order(int team_idx) {
    for (int i = 0; i < NUM_BATSMEN; i++) {
        teams[team_idx].batsmen[i] = teams[team_idx].original_batsmen[i];
        teams[team_idx].batsmen[i].runs_scored = 0;
        teams[team_idx].batsmen[i].balls_faced = 0;
        teams[team_idx].batsmen[i].fours = 0;
        teams[team_idx].batsmen[i].sixes = 0;
        teams[team_idx].batsmen[i].is_out = false;
        teams[team_idx].batsmen[i].strike_rate = 0.0;
        strcpy(teams[team_idx].batsmen[i].dismissal, "not out");
    }
    
    for (int i = 2; i < NUM_BATSMEN - 1; i++) {
        for (int j = i + 1; j < NUM_BATSMEN; j++) {
            if (teams[team_idx].batsmen[j].stay_duration < 
                teams[team_idx].batsmen[i].stay_duration) {
                Batsman temp = teams[team_idx].batsmen[i];
                teams[team_idx].batsmen[i] = teams[team_idx].batsmen[j];
                teams[team_idx].batsmen[j] = temp;
            }
        }
    }
}

void apply_batting_schedule(int team_idx, int sched_type) {
    match_state.batting_sched_type = sched_type;
    
    if (sched_type == SCHED_FCFS) {
        apply_fcfs_order(team_idx);
    } else if (sched_type == SCHED_SJF) {
        apply_sjf_order(team_idx);
    }
}

void capture_ball_context(BallContext* ctx) {
    pthread_mutex_lock(&index_mutex);
    ctx->striker_idx = striker_idx;
    ctx->non_striker_idx = non_striker_idx;
    ctx->bowler_idx = current_bowler_idx;
    strncpy(ctx->striker_name, batsmen[striker_idx].name, 49);
    ctx->striker_name[49] = '\0';
    strncpy(ctx->non_striker_name, batsmen[non_striker_idx].name, 49);
    ctx->non_striker_name[49] = '\0';
    strncpy(ctx->bowler_name, bowlers[current_bowler_idx].name, 49);
    ctx->bowler_name[49] = '\0';
    ctx->over_number = match_state.current_over;
    ctx->ball_number = match_state.current_ball;
    ctx->score_before = match_state.total_runs;
    ctx->wickets_before = match_state.wickets;
    pthread_mutex_unlock(&index_mutex);
}

int select_next_bowler_rr(void) {
    int next_bowler;
    int attempts = 0;
    int previous_bowler = current_bowler_idx;
    
    do {
        scheduler.bowler_queue_front = (scheduler.bowler_queue_front + 1) % NUM_BOWLERS;
        next_bowler = scheduler.bowler_queue[scheduler.bowler_queue_front];
        attempts++;
        
        if (bowlers[next_bowler].overs_bowled < 4 && next_bowler != previous_bowler) {
            break;
        }
    } while (attempts < NUM_BOWLERS * 2);
    
    return next_bowler;
}

int select_next_bowler_priority(void) {
    int best_bowler = -1;
    int highest_priority = 999;
    
    for (int i = 0; i < NUM_BOWLERS; i++) {
        if (bowlers[i].overs_bowled < 4 && i != current_bowler_idx) {
            int effective_priority = bowlers[i].priority;
            
            if (match_state.current_over >= DEATH_OVER_START && bowlers[i].is_death_specialist) {
                effective_priority = 0;
            }
            else if (match_state.current_over < POWERPLAY_END && bowlers[i].is_powerplay_specialist) {
                effective_priority = 0;
            }
            
            if (bowlers[i].overs_bowled > 0) {
                float eco = (float)bowlers[i].runs_given / 
                           ((float)(bowlers[i].overs_bowled * 6 + bowlers[i].balls_in_current_over) / 6.0);
                if (eco < 6.0) effective_priority--;
            }
            
            if (effective_priority < highest_priority) {
                highest_priority = effective_priority;
                best_bowler = i;
            }
        }
    }
    
    return (best_bowler == -1) ? select_next_bowler_rr() : best_bowler;
}

void print_over_summary(void) {
    pthread_mutex_lock(&log_mutex);
    
    int bowler = current_bowler_idx;
    bool maiden = (match_state.runs_in_current_over == 0 && 
                   bowlers[bowler].balls_in_current_over == 6);
    
    if (maiden) {
        bowlers[bowler].maidens++;
    }
    
    printf("\nEnd of Over %d: %s %d-%d-%d-%d %s | %s %d/%d | RR: %.2f\n",
             match_state.current_over + 1,
             bowlers[bowler].name,
             bowlers[bowler].overs_bowled + 1,
             bowlers[bowler].maidens,
             bowlers[bowler].runs_given,
             bowlers[bowler].wickets,
             maiden ? "(MAIDEN)" : "",
             match_state.team_batting,
             match_state.total_runs,
             match_state.wickets,
             match_state.current_run_rate);
    
    pthread_mutex_unlock(&log_mutex);
}

void context_switch_bowler(void) {
    pthread_mutex_lock(&scheduler_mutex);
    pthread_mutex_lock(&index_mutex);
    
    int old_bowler = current_bowler_idx;
    
    bool maiden = (match_state.runs_in_current_over == 0);
    if (maiden && bowlers[old_bowler].balls_in_current_over >= 6) {
        bowlers[old_bowler].maidens++;
    }
    
    bowlers[old_bowler].overs_bowled++;
    
    int total_balls = bowlers[old_bowler].overs_bowled * 6;
    if (total_balls > 0) {
        bowlers[old_bowler].economy = (float)bowlers[old_bowler].runs_given / 
                                       ((float)total_balls / 6.0);
    }
    
    pthread_mutex_unlock(&index_mutex);
    pthread_mutex_unlock(&scheduler_mutex);
    
    print_over_summary();
    
    pthread_mutex_lock(&scheduler_mutex);
    pthread_mutex_lock(&index_mutex);
    
    if (match_state.current_over >= DEATH_OVER_START - 1 || 
        match_state.match_intensity > HIGH_INTENSITY_THRESHOLD) {
        scheduler.scheduling_algo = 2;
        current_bowler_idx = select_next_bowler_priority();
    } else if (match_state.current_over < POWERPLAY_END) {
        scheduler.scheduling_algo = 2;
        current_bowler_idx = select_next_bowler_priority();
    } else {
        scheduler.scheduling_algo = 0;
        current_bowler_idx = select_next_bowler_rr();
    }
    
    scheduler.last_bowler = old_bowler;
    bowlers[current_bowler_idx].balls_in_current_over = 0;
    
    int temp = striker_idx;
    striker_idx = non_striker_idx;
    non_striker_idx = temp;
    crease_ends[0].occupant_id = striker_idx;
    crease_ends[1].occupant_id = non_striker_idx;
    
    match_state.current_over++;
    match_state.current_ball = 0;
    match_state.runs_in_current_over = 0;
    scheduler.current_quantum_used = 0;
    
    pthread_mutex_unlock(&index_mutex);
    pthread_mutex_unlock(&scheduler_mutex);
}

void add_edge_waitgraph(int from, int to) {
    pthread_mutex_lock(&waitgraph_mutex);
    wait_graph.adj_matrix[from][to] = 1;
    wait_graph.is_waiting[from] = true;
    pthread_mutex_unlock(&waitgraph_mutex);
}

void remove_edge_waitgraph(int from, int to) {
    pthread_mutex_lock(&waitgraph_mutex);
    wait_graph.adj_matrix[from][to] = 0;
    wait_graph.is_waiting[from] = false;
    pthread_mutex_unlock(&waitgraph_mutex);
}

bool detect_cycle_dfs(int node, bool visited[], bool rec_stack[]) {
    visited[node] = true;
    rec_stack[node] = true;
    
    for (int i = 0; i < NUM_BATSMEN; i++) {
        if (wait_graph.adj_matrix[node][i]) {
            if (!visited[i] && detect_cycle_dfs(i, visited, rec_stack)) {
                return true;
            } else if (rec_stack[i]) {
                return true;
            }
        }
    }
    
    rec_stack[node] = false;
    return false;
}

bool detect_deadlock(void) {
    pthread_mutex_lock(&waitgraph_mutex);
    
    bool visited[NUM_BATSMEN] = {false};
    bool rec_stack[NUM_BATSMEN] = {false};
    
    for (int i = 0; i < NUM_BATSMEN; i++) {
        if (wait_graph.is_waiting[i] && !visited[i]) {
            if (detect_cycle_dfs(i, visited, rec_stack)) {
                pthread_mutex_unlock(&waitgraph_mutex);
                return true;
            }
        }
    }
    
    pthread_mutex_unlock(&waitgraph_mutex);
    return false;
}

int resolve_deadlock(void) {
    printf("DEADLOCK DETECTED - Third Umpire reviewing...\n");
    
    int batsman_to_dismiss = -1;
    
    pthread_mutex_lock(&waitgraph_mutex);
    
    for (int i = 0; i < NUM_BATSMEN; i++) {
        if (wait_graph.is_waiting[i]) {
            if (batsman_to_dismiss == -1 || 
                batsmen[i].runs_scored < batsmen[batsman_to_dismiss].runs_scored) {
                batsman_to_dismiss = i;
            }
        }
    }
    
    for (int i = 0; i < NUM_BATSMEN; i++) {
        wait_graph.is_waiting[i] = false;
        wait_graph.waiting_for_resource[i] = -1;
        for (int j = 0; j < NUM_BATSMEN; j++) {
            wait_graph.adj_matrix[i][j] = 0;
        }
    }
    
    pthread_mutex_unlock(&waitgraph_mutex);
    
    if (batsman_to_dismiss != -1) {
        batsmen[batsman_to_dismiss].is_out = true;
        snprintf(batsmen[batsman_to_dismiss].dismissal, 100, 
                 "run out (Deadlock Resolution)");
        printf("OUT! %s RUN OUT (Deadlock resolved)\n", batsmen[batsman_to_dismiss].name);
    }
    
    return batsman_to_dismiss;
}

void handle_runout_scenario(const BallContext* ctx) {
    int runs_attempted = pitch_ball.runs_scored;
    
    if (runs_attempted <= 0 || runs_attempted >= 4) return;
    
    pthread_mutex_lock(&index_mutex);
    int current_striker = striker_idx;
    int current_non_striker = non_striker_idx;
    pthread_mutex_unlock(&index_mutex);
    
    for (int run = 0; run < runs_attempted; run++) {
        add_edge_waitgraph(current_striker, current_non_striker);
        add_edge_waitgraph(current_non_striker, current_striker);
        
        if (detect_deadlock()) {
            int runout_prob = 12 + (match_state.match_intensity / 10);
            
            if (rand() % 100 < runout_prob) {
                int dismissed = resolve_deadlock();
                if (dismissed != -1) {
                    pthread_mutex_lock(&score_mutex);
                    match_state.wickets++;
                    pthread_mutex_unlock(&score_mutex);
                    
                    pitch_ball.is_wicket = true;
                    pitch_ball.wicket_type = 4;
                    pitch_ball.dismissed_batsman_idx = dismissed;
                    strncpy(pitch_ball.dismissed_batsman_name, 
                            batsmen[dismissed].name, 49);
                    
                    new_batsman();
                    break;
                }
            }
        }
        
        remove_edge_waitgraph(current_striker, current_non_striker);
        remove_edge_waitgraph(current_non_striker, current_striker);
    }
    
    if (runs_attempted % 2 == 1 && !pitch_ball.is_wicket) {
        pthread_mutex_lock(&index_mutex);
        int temp = striker_idx;
        striker_idx = non_striker_idx;
        non_striker_idx = temp;
        pthread_mutex_unlock(&index_mutex);
    }
}

int generate_ball_outcome(const BallContext* ctx) {
    int rand_val = rand() % 1000;
    int striker = ctx->striker_idx;
    int bowler = ctx->bowler_idx;
    
    int bat_skill = 700 - (batsmen[striker].is_tail_ender ? 250 : 0);
    int bowl_skill = 600 + (bowlers[bowler].is_death_specialist ? 80 : 0);
    
    if (match_state.match_intensity > 70) bat_skill += 50;
    if (match_state.current_over >= DEATH_OVER_START) {
        bat_skill += 30;
        bowl_skill += 20;
    }
    if (match_state.current_over < POWERPLAY_END) {
        bat_skill += 40;
    }
    
    if (!match_state.is_first_innings && match_state.target > 0) {
        int needed = match_state.target - match_state.total_runs;
        if (match_state.required_run_rate > 12.0) {
            bat_skill += 60;
        }
        if (needed <= 20 && match_state.balls_remaining <= 12) {
            bat_skill += 80;
        }
    }
    
    if (rand_val < 25) return OUTCOME_WIDE;
    if (rand_val < 40) return OUTCOME_NOBALL;
    
    int wicket_prob = 50 + (bowl_skill - bat_skill) / 8;
    if (wicket_prob < 30) wicket_prob = 30;
    if (wicket_prob > 120) wicket_prob = 120;
    
    if (rand_val < 40 + wicket_prob) return OUTCOME_WICKET;
    
    int boundary_boost = (match_state.current_over >= DEATH_OVER_START) ? 30 : 0;
    int powerplay_boost = (match_state.current_over < POWERPLAY_END) ? 25 : 0;
    
    if (rand_val < 340) return OUTCOME_DOT;
    if (rand_val < 590) return OUTCOME_SINGLE;
    if (rand_val < 710) return OUTCOME_DOUBLE;
    if (rand_val < 740) return OUTCOME_TRIPLE;
    if (rand_val < 890 + boundary_boost + powerplay_boost) return OUTCOME_FOUR;
    return OUTCOME_SIX;
}

void process_ball_outcome(int outcome, const BallContext* ctx) {
    pthread_mutex_lock(&score_mutex);
    
    int striker = ctx->striker_idx;
    int bowler = ctx->bowler_idx;
    int runs = 0;
    bool wicket = false;
    bool extra = false;
    bool legal_delivery = true;
    
    pitch_ball.dismissed_batsman_idx = -1;
    strcpy(pitch_ball.dismissed_batsman_name, "");
    strcpy(pitch_ball.outcome_str, "");
    
    switch (outcome) {
        case OUTCOME_DOT:
            runs = 0;
            strcpy(pitch_ball.outcome_str, ".");
            bowlers[bowler].dots++;
            break;
            
        case OUTCOME_SINGLE:
            runs = 1;
            strcpy(pitch_ball.outcome_str, "1");
            break;
            
        case OUTCOME_DOUBLE:
            runs = 2;
            strcpy(pitch_ball.outcome_str, "2");
            break;
            
        case OUTCOME_TRIPLE:
            runs = 3;
            strcpy(pitch_ball.outcome_str, "3");
            break;
            
        case OUTCOME_FOUR:
            runs = 4;
            strcpy(pitch_ball.outcome_str, "4");
            batsmen[striker].fours++;
            total_boundaries++;
            break;
            
        case OUTCOME_SIX:
            runs = 6;
            strcpy(pitch_ball.outcome_str, "6");
            batsmen[striker].sixes++;
            total_sixes++;
            ball_in_air = true;
            break;
            
        case OUTCOME_WICKET: {
            runs = 0;
            wicket = true;
            int wicket_type = rand() % 5;
            pitch_ball.wicket_type = wicket_type;
            pitch_ball.dismissed_batsman_idx = striker;
            strncpy(pitch_ball.dismissed_batsman_name, ctx->striker_name, 49);
            
            const char* wkt_types[] = {"b", "c", "lbw", "st", "c&b"};
            strcpy(pitch_ball.outcome_str, "W");
            
            if (wicket_type == 1 || wicket_type == 4) ball_in_air = true;
            
            const char* dismissal_formats[] = {
                "b %s", "c sub b %s", "lbw b %s", "st sub b %s", "c & b %s"
            };
            snprintf(batsmen[striker].dismissal, 100, dismissal_formats[wicket_type], ctx->bowler_name);
            batsmen[striker].is_out = true;
            break;
        }
            
        case OUTCOME_WIDE:
            runs = 1;
            extra = true;
            legal_delivery = false;
            strcpy(pitch_ball.outcome_str, "Wd");
            bowlers[bowler].wides++;
            match_state.wides++;
            pitch_ball.is_wide = true;
            break;
            
        case OUTCOME_NOBALL:
            runs = 1;
            extra = true;
            legal_delivery = false;
            strcpy(pitch_ball.outcome_str, "Nb");
            bowlers[bowler].noballs++;
            match_state.noballs++;
            pitch_ball.is_noball = true;
            break;
            
        case OUTCOME_LEG_BYE:
            runs = 1;
            extra = true;
            strcpy(pitch_ball.outcome_str, "lb");
            match_state.leg_byes++;
            pitch_ball.is_leg_bye = true;
            break;
    }
    
    match_state.total_runs += runs;
    match_state.runs_in_current_over += runs;
    
    if (!extra) {
        batsmen[striker].runs_scored += runs;
        if (legal_delivery) {
            batsmen[striker].balls_faced++;
            if (batsmen[striker].balls_faced > 0) {
                batsmen[striker].strike_rate = 
                    (float)batsmen[striker].runs_scored * 100.0 / (float)batsmen[striker].balls_faced;
            }
        }
    } else {
        match_state.extras += runs;
    }
    
    bowlers[bowler].runs_given += runs;
    pitch_ball.runs_scored = runs;
    pitch_ball.is_wicket = wicket;
    
    snprintf(pitch_ball.description, sizeof(pitch_ball.description),
             "%d.%d %s to %s, %s",
             ctx->over_number, ctx->ball_number + 1,
             ctx->bowler_name, ctx->striker_name, pitch_ball.outcome_str);
    
    if (wicket) {
        match_state.wickets++;
        bowlers[bowler].wickets++;
    }
    
    int total_balls_by_bowler = bowlers[bowler].overs_bowled * 6 + bowlers[bowler].balls_in_current_over + 1;
    if (total_balls_by_bowler > 0) {
        bowlers[bowler].economy = (float)bowlers[bowler].runs_given / ((float)total_balls_by_bowler / 6.0);
    }
    
    memcpy(&pitch_ball.context, ctx, sizeof(BallContext));
    
    pthread_mutex_unlock(&score_mutex);
}

void switch_strike(void) {
    pthread_mutex_lock(&index_mutex);
    int temp = striker_idx;
    striker_idx = non_striker_idx;
    non_striker_idx = temp;
    crease_ends[0].occupant_id = striker_idx;
    crease_ends[1].occupant_id = non_striker_idx;
    pthread_mutex_unlock(&index_mutex);
}

void new_batsman(void) {
    pthread_mutex_lock(&index_mutex);
    
    int new_batsman_idx = next_batsman_idx;
    
    while (new_batsman_idx < NUM_BATSMEN && batsmen[new_batsman_idx].is_out) {
        new_batsman_idx++;
    }
    
    if (new_batsman_idx < NUM_BATSMEN && !batsmen[new_batsman_idx].is_out) {
        if (batsmen[striker_idx].is_out) {
            striker_idx = new_batsman_idx;
        } else if (batsmen[non_striker_idx].is_out) {
            non_striker_idx = new_batsman_idx;
        }
        
        next_batsman_idx = new_batsman_idx + 1;
        
        crease_ends[0].occupant_id = striker_idx;
        crease_ends[1].occupant_id = non_striker_idx;
        
        printf("New batsman: %s\n", batsmen[new_batsman_idx].name);
    }
    
    pthread_mutex_unlock(&index_mutex);
}

void calculate_match_intensity(void) {
    int intensity = 0;
    
    if (match_state.current_over >= DEATH_OVER_START) intensity += 30;
    if (match_state.current_over >= MAX_OVERS - 2) intensity += 25;
    if (match_state.current_over == MAX_OVERS - 1) intensity += 20;
    if (match_state.wickets >= 5) intensity += 15;
    if (match_state.wickets >= 7) intensity += 15;
    if (match_state.wickets >= 9) intensity += 20;
    
    if (!match_state.is_first_innings && match_state.target > 0) {
        int remaining = match_state.target - match_state.total_runs;
        int balls_left = (MAX_OVERS - match_state.current_over) * 6 - match_state.current_ball;
        
        if (balls_left > 0) {
            float rrr = (float)remaining * 6.0 / (float)balls_left;
            if (rrr > 10.0) intensity += 25;
            if (rrr > 12.0) intensity += 15;
        }
        
        if (remaining <= 30 && balls_left <= 18) intensity += 20;
        if (remaining <= 15 && balls_left <= 12) intensity += 25;
    }
    
    match_state.match_intensity = (intensity > 100) ? 100 : intensity;
}

void update_run_rates(void) {
    int total_balls = match_state.current_over * 6 + match_state.current_ball;
    
    if (total_balls > 0) {
        match_state.current_run_rate = (float)match_state.total_runs * 6.0 / (float)total_balls;
    }
    
    match_state.balls_remaining = (MAX_OVERS * 6) - total_balls;
    
    if (!match_state.is_first_innings && match_state.target > 0 && match_state.balls_remaining > 0) {
        int needed = match_state.target - match_state.total_runs;
        match_state.required_run_rate = (float)needed * 6.0 / (float)match_state.balls_remaining;
    }
}

void* bowler_thread_func(void* arg) {
    int bowler_id = *(int*)arg;
    
    while (match_running) {
        pthread_mutex_lock(&scheduler_mutex);
        int active_bowler = current_bowler_idx;
        pthread_mutex_unlock(&scheduler_mutex);
        
        if (active_bowler != bowler_id) {
            usleep(30000);
            continue;
        }
        
        if (bowlers[bowler_id].overs_bowled >= 4) {
            context_switch_bowler();
            continue;
        }
        
        BallContext ctx;
        capture_ball_context(&ctx);
        
        pthread_mutex_lock(&pitch_mutex);
        pitch_ball.ball_number++;
        pitch_ball.is_bowled = true;
        pitch_ball.is_played = false;
        pitch_ball.ball_in_air = false;
        pitch_ball.is_wide = false;
        pitch_ball.is_noball = false;
        pitch_ball.is_wicket = false;
        pitch_ball.is_leg_bye = false;
        pitch_ball.dismissed_batsman_idx = -1;
        memcpy(&pitch_ball.context, &ctx, sizeof(BallContext));
        pthread_mutex_unlock(&pitch_mutex);
        
        sem_post(&ball_ready_sem);
        sem_wait(&stroke_complete_sem);
        
        pthread_mutex_lock(&scheduler_mutex);
        if (!pitch_ball.is_wide && !pitch_ball.is_noball) {
            bowlers[bowler_id].balls_in_current_over++;
            match_state.current_ball++;
            scheduler.current_quantum_used++;
            total_balls_bowled++;
        }
        pthread_mutex_unlock(&scheduler_mutex);
        
        print_ball_log(&pitch_ball.context);
        update_run_rates();
        
        if (scheduler.current_quantum_used >= TIME_QUANTUM) {
            context_switch_bowler();
        }
        
        if (match_state.wickets >= MAX_WICKETS || match_state.current_over >= MAX_OVERS) {
            innings_over = true;
            match_running = false;
        }
        
        if (!match_state.is_first_innings && match_state.total_runs >= match_state.target) {
            innings_over = true;
            match_running = false;
        }
        
        usleep(50000);
    }
    return NULL;
}

void* batsman_thread_func(void* arg) {
    int batsman_id = *(int*)arg;
    
    while (match_running) {
        pthread_mutex_lock(&index_mutex);
        bool is_striker = (batsman_id == striker_idx);
        bool is_non_striker = (batsman_id == non_striker_idx);
        pthread_mutex_unlock(&index_mutex);
        
        if (!is_striker && !is_non_striker) {
            usleep(30000);
            continue;
        }
        
        if (!is_striker) {
            usleep(15000);
            continue;
        }
        
        sem_wait(&ball_ready_sem);
        if (!match_running) break;
        
        BallContext ctx;
        pthread_mutex_lock(&pitch_mutex);
        memcpy(&ctx, &pitch_ball.context, sizeof(BallContext));
        pthread_mutex_unlock(&pitch_mutex);
        
        if (ctx.striker_idx != batsman_id) {
            sem_post(&stroke_complete_sem);
            continue;
        }
        
        pthread_mutex_lock(&pitch_mutex);
        if (pitch_ball.is_bowled && !pitch_ball.is_played) {
            int outcome = generate_ball_outcome(&ctx);
            process_ball_outcome(outcome, &ctx);
            pitch_ball.is_played = true;
            
            if (ball_in_air) {
                pthread_mutex_lock(&ball_state_mutex);
                pthread_cond_broadcast(&ball_hit_cond);
                pthread_mutex_unlock(&ball_state_mutex);
            }
        }
        pthread_mutex_unlock(&pitch_mutex);
        
        if (pitch_ball.runs_scored > 0 && pitch_ball.runs_scored < 4 && !pitch_ball.is_wicket) {
            handle_runout_scenario(&ctx);
        } else if (pitch_ball.runs_scored > 0 && !pitch_ball.is_wicket) {
            if (pitch_ball.runs_scored % 2 == 1) switch_strike();
        }
        
        if (pitch_ball.is_wicket && match_state.wickets < MAX_WICKETS) {
            new_batsman();
        }
        
        calculate_match_intensity();
        sem_post(&stroke_complete_sem);
    }
    return NULL;
}

void* fielder_thread_func(void* arg) {
    int fielder_id = *(int*)arg;
    
    while (match_running) {
        pthread_mutex_lock(&ball_state_mutex);
        while (!ball_in_air && match_running) {
            pthread_cond_wait(&ball_hit_cond, &ball_state_mutex);
        }
        if (!match_running) {
            pthread_mutex_unlock(&ball_state_mutex);
            break;
        }
        fielders[fielder_id].is_active = true;
        pthread_mutex_unlock(&ball_state_mutex);
        
        usleep(30000);
        
        if (pitch_ball.is_wicket && (pitch_ball.wicket_type == 1 || pitch_ball.wicket_type == 4)) {
            if (rand() % 100 < 75) {
                fielders[fielder_id].catches++;
            }
        }
        
        pthread_mutex_lock(&ball_state_mutex);
        ball_in_air = false;
        fielders[fielder_id].is_active = false;
        pthread_mutex_unlock(&ball_state_mutex);
    }
    return NULL;
}

void* umpire_thread_func(void* arg) {
    (void)arg;
    
    while (match_running) {
        if (detect_deadlock()) {
            resolve_deadlock();
        }
        
        pthread_mutex_lock(&match_mutex);
        
        if (match_state.wickets >= MAX_WICKETS) {
            printf("\nALL OUT!\n");
            match_running = false;
        }
        
        if (match_state.current_over >= MAX_OVERS) {
            printf("\n20 OVERS COMPLETE!\n");
            match_running = false;
        }
        
        if (!match_state.is_first_innings && match_state.total_runs >= match_state.target) {
            printf("\nTARGET ACHIEVED!\n");
            match_running = false;
        }
        
        pthread_mutex_unlock(&match_mutex);
        
        usleep(80000);
    }
    return NULL;
}

void print_ball_log(const BallContext* ctx) {
    pthread_mutex_lock(&log_mutex);
    
    printf("%s", pitch_ball.description);
    
    if (pitch_ball.is_wicket) {
        int dismissed_idx = pitch_ball.dismissed_batsman_idx >= 0 ? 
                            pitch_ball.dismissed_batsman_idx : ctx->striker_idx;
        printf(" - OUT! %s %d(%d)",
               batsmen[dismissed_idx].name,
               batsmen[dismissed_idx].runs_scored,
               batsmen[dismissed_idx].balls_faced);
    }
    
    if (pitch_ball.runs_scored == 4) {
        printf(" FOUR!");
    } else if (pitch_ball.runs_scored == 6) {
        printf(" SIX!");
    }
    
    printf("\n");
    fflush(stdout);
    pthread_mutex_unlock(&log_mutex);
}

void print_scorecard(void) {
    printf("\nSCORECARD - %s (%s)\n", match_state.team_batting,
           match_state.batting_sched_type == SCHED_FCFS ? "FCFS" : "SJF");
    printf("%-25s %-25s %4s %4s %3s %3s %6s\n", 
           "BATSMAN", "DISMISSAL", "R", "B", "4", "6", "SR");
    printf("----------------------------------------------------------------------\n");
    
    for (int i = 0; i < NUM_BATSMEN; i++) {
        if (batsmen[i].balls_faced > 0 || batsmen[i].is_out) {
            bool is_current = (i == striker_idx || i == non_striker_idx) && !batsmen[i].is_out;
            float sr = batsmen[i].balls_faced > 0 ? 
                       (float)batsmen[i].runs_scored * 100.0 / batsmen[i].balls_faced : 0.0;
            
            printf("%-25s %-25s %4d %4d %3d %3d %6.1f\n",
                   batsmen[i].name,
                   batsmen[i].is_out ? batsmen[i].dismissal : (is_current ? "not out *" : "not out"),
                   batsmen[i].runs_scored,
                   batsmen[i].balls_faced,
                   batsmen[i].fours,
                   batsmen[i].sixes,
                   sr);
        }
    }
    
    printf("----------------------------------------------------------------------\n");
    printf("Extras: (w %d, nb %d, lb %d) %d\n", 
           match_state.wides, match_state.noballs, match_state.leg_byes, match_state.extras);
    printf("TOTAL: %d/%d (%d.%d overs) RR: %.2f\n\n",
           match_state.total_runs, match_state.wickets,
           match_state.current_over, match_state.current_ball,
           match_state.current_run_rate);
    
    printf("BOWLING - %s\n", match_state.team_bowling);
    printf("%-25s %5s %5s %5s %5s %5s %8s\n", 
           "BOWLER", "O", "M", "R", "W", "DOT", "ECON");
    printf("----------------------------------------------------------------------\n");
    
    for (int i = 0; i < NUM_BOWLERS; i++) {
        int total_balls = bowlers[i].overs_bowled * 6 + bowlers[i].balls_in_current_over;
        if (total_balls > 0) {
            float economy = (float)bowlers[i].runs_given / ((float)total_balls / 6.0);
            printf("%-25s %3d.%d %5d %5d %5d %5d %8.2f\n",
                   bowlers[i].name,
                   bowlers[i].overs_bowled,
                   bowlers[i].balls_in_current_over,
                   bowlers[i].maidens,
                   bowlers[i].runs_given,
                   bowlers[i].wickets,
                   bowlers[i].dots,
                   economy);
        }
    }
    printf("----------------------------------------------------------------------\n");
}

void print_final_result(void) {
    printf("\n======================================================================\n");
    printf("                           MATCH RESULT\n");
    printf("======================================================================\n");
    printf("%s: %d/%d (%d.%d overs) RR: %.2f\n",
           teams[0].name, teams[0].innings_score, teams[0].innings_wickets,
           teams[0].innings_overs, teams[0].innings_balls,
           teams[0].innings_run_rate);
    printf("%s: %d/%d (%d.%d overs) RR: %.2f\n",
           teams[1].name, teams[1].innings_score, teams[1].innings_wickets,
           teams[1].innings_overs, teams[1].innings_balls,
           teams[1].innings_run_rate);
    printf("----------------------------------------------------------------------\n");
    
    if (teams[1].innings_score > teams[0].innings_score) {
        int wickets_remaining = MAX_WICKETS - teams[1].innings_wickets;
        int balls_remaining = (MAX_OVERS * 6) - (teams[1].innings_overs * 6 + teams[1].innings_balls);
        printf("%s WINS by %d wickets (with %d balls remaining)\n",
               teams[1].name, wickets_remaining, balls_remaining);
    } else if (teams[0].innings_score > teams[1].innings_score) {
        int margin = teams[0].innings_score - teams[1].innings_score;
        printf("%s WINS by %d runs\n", teams[0].name, margin);
    } else {
        printf("MATCH TIED\n");
    }
    printf("======================================================================\n");
}

void cleanup_resources(void) {
    pthread_mutex_destroy(&pitch_mutex);
    pthread_mutex_destroy(&score_mutex);
    pthread_mutex_destroy(&match_mutex);
    pthread_mutex_destroy(&ball_state_mutex);
    pthread_mutex_destroy(&scheduler_mutex);
    pthread_mutex_destroy(&waitgraph_mutex);
    pthread_mutex_destroy(&log_mutex);
    pthread_mutex_destroy(&index_mutex);
    
    sem_destroy(&crease_semaphore);
    sem_destroy(&ball_ready_sem);
    sem_destroy(&stroke_complete_sem);
    sem_destroy(&over_complete_sem);
    
    pthread_cond_destroy(&ball_hit_cond);
    pthread_cond_destroy(&ball_bowled_cond);
    pthread_cond_destroy(&fielder_wake_cond);
    
    for (int i = 0; i < 2; i++) {
        pthread_mutex_destroy(&crease_ends[i].lock);
    }
}

int get_scheduling_choice(int innings_num) {
    int choice;
    int bat_team = (innings_num == 1) ? 0 : 1;
    
    printf("\nINNINGS %d - %s BATTING\n", innings_num, teams[bat_team].name);
    printf("Select batting order scheduling:\n");
    printf("1. FCFS (First Come First Serve - Original order)\n");
    printf("2. SJF (Shortest Job First - Tail-enders bat early)\n");
    printf("Enter choice (1 or 2): ");
    
    if (scanf("%d", &choice) != 1 || (choice != 1 && choice != 2)) {
        printf("Invalid choice. Defaulting to FCFS.\n");
        choice = 1;
        while (getchar() != '\n');
    }
    
    return choice;
}

void simulate_innings(int innings_num) {
    int bat_team = (innings_num == 1) ? 0 : 1;
    int bowl_team = (innings_num == 1) ? 1 : 0;
    
    int sched_choice = get_scheduling_choice(innings_num);
    
    printf("\nINNINGS %d: %s vs %s\n", innings_num, teams[bat_team].name, teams[bowl_team].name);
    printf("Scheduling: %s\n", sched_choice == SCHED_FCFS ? "FCFS" : "SJF");
    if (innings_num == 2) {
        printf("Target: %d\n", match_state.target);
    }
    
    reset_for_innings(bat_team, bowl_team);
    apply_batting_schedule(bat_team, sched_choice);
    
    match_state.is_first_innings = (innings_num == 1);
    
    printf("\nPress ENTER to start...");
    while (getchar() != '\n');
    getchar();
    printf("\n");
    
    pthread_t bowler_threads[NUM_BOWLERS];
    pthread_t batsman_threads[NUM_BATSMEN];
    pthread_t fielder_threads[NUM_FIELDERS];
    pthread_t umpire_thread;
    
    int bowler_ids[NUM_BOWLERS];
    int batsman_ids[NUM_BATSMEN];
    int fielder_ids[NUM_FIELDERS];
    
    pthread_create(&umpire_thread, NULL, umpire_thread_func, NULL);
    
    for (int i = 0; i < NUM_BOWLERS; i++) {
        bowler_ids[i] = i;
        pthread_create(&bowler_threads[i], NULL, bowler_thread_func, &bowler_ids[i]);
    }
    
    for (int i = 0; i < NUM_BATSMEN; i++) {
        batsman_ids[i] = i;
        pthread_create(&batsman_threads[i], NULL, batsman_thread_func, &batsman_ids[i]);
    }
    
    for (int i = 0; i < NUM_FIELDERS; i++) {
        fielder_ids[i] = i;
        pthread_create(&fielder_threads[i], NULL, fielder_thread_func, &fielder_ids[i]);
    }
    
    pthread_join(umpire_thread, NULL);
    
    match_running = false;
    
    for (int i = 0; i < 5; i++) {
        sem_post(&ball_ready_sem);
        sem_post(&stroke_complete_sem);
    }
    pthread_cond_broadcast(&ball_hit_cond);
    
    for (int i = 0; i < NUM_BOWLERS; i++) pthread_cancel(bowler_threads[i]);
    for (int i = 0; i < NUM_BATSMEN; i++) pthread_cancel(batsman_threads[i]);
    for (int i = 0; i < NUM_FIELDERS; i++) pthread_cancel(fielder_threads[i]);
    
    usleep(100000);
    
    int total_balls = match_state.current_over * 6 + match_state.current_ball;
    teams[bat_team].innings_score = match_state.total_runs;
    teams[bat_team].innings_wickets = match_state.wickets;
    teams[bat_team].innings_overs = match_state.current_over;
    teams[bat_team].innings_balls = match_state.current_ball;
    teams[bat_team].innings_extras = match_state.extras;
    teams[bat_team].innings_run_rate = total_balls > 0 ? 
                                        (float)match_state.total_runs * 6.0 / total_balls : 0.0;
    
    print_scorecard();
    
    if (innings_num == 1) {
        match_state.target = teams[bat_team].innings_score + 1;
    }
}

int main(void) {
    printf("T20 CRICKET SIMULATOR\n");
    printf("INDIA vs AUSTRALIA - 20 Overs Per Innings\n\n");
    
    srand(time(NULL));
    
    initialize_teams();
    initialize_match_state();
    initialize_scheduler();
    initialize_synchronization();
    initialize_wait_graph();
    
    simulate_innings(1);
    
    printf("\nINNINGS BREAK - %s need %d runs to win\n", teams[1].name, match_state.target);
    
    simulate_innings(2);
    
    print_final_result();
    cleanup_resources();
    
    return 0;
}