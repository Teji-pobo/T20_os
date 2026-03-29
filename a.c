/*
 * T20 Cricket World Cup 2026 Simulator - SJF vs FCFS VERSION
 * Multi-threaded simulation using pthreads
 * 
 * Features:
 * - Full 20 overs per innings
 * - Two complete innings simulation
 * - SJF and FCFS batting order scheduling
 * - Round Robin and Priority bowler scheduling
 * - Deadlock detection for run-out scenarios
 * - Winner declaration with proper margin calculation
 * 
 * Scheduling Options:
 * - SJF (Shortest Job First): Tail-enders bat early based on expected stay
 * - FCFS (First Come First Serve): Original batting order maintained
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>

/*============================================================================
 * CONSTANTS AND MACROS
 *============================================================================*/
#define MAX_OVERS 20          // Full T20 match - 20 overs per innings
#define BALLS_PER_OVER 6
#define MAX_WICKETS 10
#define NUM_FIELDERS 10
#define NUM_BOWLERS 5
#define NUM_BATSMEN 11
#define TIME_QUANTUM 6        // Bowler rotation every 6 balls (1 over)
#define CREASE_CAPACITY 2
#define DEATH_OVER_START 16   // Death overs: 16-20
#define POWERPLAY_END 6       // Powerplay: 1-6
#define HIGH_INTENSITY_THRESHOLD 70

// Scheduling types for batsmen
#define SCHED_FCFS 1          // First Come First Serve (Original Order)
#define SCHED_SJF 2           // Shortest Job First

// Outcome types
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

// Colors for terminal output
#define RED "\033[1;31m"
#define GREEN "\033[1;32m"
#define YELLOW "\033[1;33m"
#define BLUE "\033[1;34m"
#define MAGENTA "\033[1;35m"
#define CYAN "\033[1;36m"
#define WHITE "\033[1;37m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

/*============================================================================
 * DATA STRUCTURES
 *============================================================================*/

typedef struct {
    int id;
    char name[50];
    int runs_scored;
    int balls_faced;
    int fours;
    int sixes;
    bool is_out;
    int stay_duration;        // Expected stay for SJF (in balls)
    int priority;             // 1=Top order, 2=Middle, 3=Lower
    bool is_tail_ender;
    char dismissal[100];
    int original_order;       // Original batting position for FCFS
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
    int batting_sched_type;   // FCFS or SJF
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
    Batsman original_batsmen[NUM_BATSMEN];  // Store original order for FCFS
    Bowler bowlers[NUM_BOWLERS];
    Fielder fielders[NUM_FIELDERS];
    int innings_score;
    int innings_wickets;
    int innings_overs;
    int innings_balls;
    int innings_extras;
    float innings_run_rate;
} Team;

/*============================================================================
 * GLOBAL VARIABLES
 *============================================================================*/

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

// Synchronization Primitives
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

FILE* match_log;

// Statistics
int total_balls_bowled = 0;
int total_boundaries = 0;
int total_sixes = 0;

/*============================================================================
 * FUNCTION PROTOTYPES
 *============================================================================*/

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
const char* generate_commentary(int outcome, const BallContext* ctx);

void* bowler_thread_func(void* arg);
void* batsman_thread_func(void* arg);
void* fielder_thread_func(void* arg);
void* umpire_thread_func(void* arg);
void* scoreboard_thread_func(void* arg);
void* third_umpire_thread_func(void* arg);

void add_edge_waitgraph(int from, int to);
void remove_edge_waitgraph(int from, int to);
bool detect_cycle_dfs(int node, bool visited[], bool rec_stack[]);
bool detect_deadlock(void);
int resolve_deadlock(void);
void handle_runout_scenario(const BallContext* ctx);

void print_ball_log(const BallContext* ctx);
void print_over_summary(void);
void print_scorecard(void);
void print_innings_summary(int team_idx);
void print_final_result(void);
void print_match_summary(void);
void log_event(const char* event);
void cleanup_resources(void);

void simulate_innings(int innings_num);
int get_scheduling_choice(int innings_num);

/*============================================================================
 * INITIALIZATION FUNCTIONS
 *============================================================================*/

void initialize_teams(void) {
    // Team 1 - INDIA (T20 WC 2026 Projected Squad)
    strcpy(teams[0].name, "INDIA");
    strcpy(teams[0].short_name, "IND");
    
    const char* india_batsmen[] = {
        "Rohit Sharma", "Yashasvi Jaiswal", "Virat Kohli", "Suryakumar Yadav",
        "Hardik Pandya", "Rishabh Pant", "Ravindra Jadeja", "Axar Patel",
        "Kuldeep Yadav", "Jasprit Bumrah", "Mohammed Siraj"
    };
    // Expected stay duration in balls for each batsman
    const int india_stay[] = {35, 30, 40, 28, 25, 22, 18, 15, 8, 5, 4};
    
    const char* india_bowlers[] = {
        "Jasprit Bumrah", "Mohammed Siraj", "Arshdeep Singh",
        "Kuldeep Yadav", "Ravindra Jadeja"
    };
    
    // Team 2 - AUSTRALIA (T20 WC 2026 Projected Squad)
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
        // Initialize batsmen
        for (int i = 0; i < NUM_BATSMEN; i++) {
            teams[t].batsmen[i].id = i;
            teams[t].batsmen[i].original_order = i;  // Store original position
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
            
            // Store original batsmen order for FCFS restoration
            teams[t].original_batsmen[i] = teams[t].batsmen[i];
        }
        
        // Initialize bowlers
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
        
        // Initialize fielders
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
    
    printf(GREEN "✓ Teams initialized successfully\n" RESET);
    printf("  Team 1: %s\n", teams[0].name);
    printf("  Team 2: %s\n", teams[1].name);
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
    
    // Reset batting team stats
    for (int i = 0; i < NUM_BATSMEN; i++) {
        batsmen[i].runs_scored = 0;
        batsmen[i].balls_faced = 0;
        batsmen[i].fours = 0;
        batsmen[i].sixes = 0;
        batsmen[i].is_out = false;
        batsmen[i].strike_rate = 0.0;
        strcpy(batsmen[i].dismissal, "not out");
    }
    
    // Reset bowling team stats
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
    
    // Reset indices
    striker_idx = 0;
    non_striker_idx = 1;
    current_bowler_idx = 0;
    next_batsman_idx = 2;
    
    // Reset match state
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
    
    // Reset scheduler
    scheduler.current_quantum_used = 0;
    scheduler.bowler_queue_front = 0;
    scheduler.last_bowler = -1;
    
    // Reset crease
    crease_ends[0].occupant_id = striker_idx;
    crease_ends[1].occupant_id = non_striker_idx;
    
    // Reset wait graph
    initialize_wait_graph();
    wait_graph.holds_resource[striker_idx] = 0;
    wait_graph.holds_resource[non_striker_idx] = 1;
    
    // Reset semaphores
    int val;
    sem_getvalue(&ball_ready_sem, &val);
    while (val > 0) { sem_wait(&ball_ready_sem); sem_getvalue(&ball_ready_sem, &val); }
    sem_getvalue(&stroke_complete_sem, &val);
    while (val > 0) { sem_wait(&stroke_complete_sem); sem_getvalue(&stroke_complete_sem, &val); }
    
    // Reset pitch
    memset(&pitch_ball, 0, sizeof(Ball));
    pitch_ball.dismissed_batsman_idx = -1;
    
    // Reset flags
    match_running = true;
    innings_over = false;
    ball_in_air = false;
    
    total_balls_bowled = 0;
}

/*============================================================================
 * BATTING ORDER SCHEDULING (FCFS / SJF)
 *============================================================================*/

// FCFS: First Come First Serve - Maintain original batting order
void apply_fcfs_order(int team_idx) {
    printf(CYAN "\n  [FCFS SCHEDULING] First Come First Serve - Original batting order\n" RESET);
    printf("  Batsmen will bat in their designated positions (1-11)\n");
    printf("  No reordering - arrival order determines batting sequence\n\n");
    
    // Restore original order from stored copy
    for (int i = 0; i < NUM_BATSMEN; i++) {
        teams[team_idx].batsmen[i] = teams[team_idx].original_batsmen[i];
        // Reset stats
        teams[team_idx].batsmen[i].runs_scored = 0;
        teams[team_idx].batsmen[i].balls_faced = 0;
        teams[team_idx].batsmen[i].fours = 0;
        teams[team_idx].batsmen[i].sixes = 0;
        teams[team_idx].batsmen[i].is_out = false;
        teams[team_idx].batsmen[i].strike_rate = 0.0;
        strcpy(teams[team_idx].batsmen[i].dismissal, "not out");
    }
    
    printf("  " YELLOW "Batting Order (FCFS - Original):" RESET "\n");
    printf("  ┌─────┬──────────────────────────┬──────────┬───────────────┐\n");
    printf("  │ No. │ Batsman                  │ Position │ Expected Stay │\n");
    printf("  ├─────┼──────────────────────────┼──────────┼───────────────┤\n");
    for (int i = 0; i < NUM_BATSMEN; i++) {
        const char* pos = (i < 2) ? "Opener" : 
                         (i < 4) ? "Top Order" :
                         (i < 7) ? "Middle Order" : "Tail-ender";
        printf("  │ %2d. │ %-24s │ %-8s │ %5d balls   │\n", 
               i + 1, 
               teams[team_idx].batsmen[i].name, 
               pos,
               teams[team_idx].batsmen[i].stay_duration);
    }
    printf("  └─────┴──────────────────────────┴──────────┴───────────────┘\n\n");
}

// SJF: Shortest Job First - Sort by expected stay duration
void apply_sjf_order(int team_idx) {
    printf(CYAN "\n  [SJF SCHEDULING] Shortest Job First - Sort by expected stay\n" RESET);
    printf("  Tail-enders (shorter stay) will bat before middle order\n");
    printf("  Openers remain unchanged (positions 1-2)\n\n");
    
    // First restore original order
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
    
    // Keep openers (0,1) in place, sort rest by expected stay (ascending)
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
    
    printf("  " YELLOW "Batting Order (SJF - Sorted by Expected Stay):" RESET "\n");
    printf("  ┌─────┬──────────────────────────┬───────────────┬──────────────────┐\n");
    printf("  │ No. │ Batsman                  │ Expected Stay │ Original Position│\n");
    printf("  ├─────┼──────────────────────────┼───────────────┼──────────────────┤\n");
    for (int i = 0; i < NUM_BATSMEN; i++) {
        printf("  │ %2d. │ %-24s │ %5d balls   │     #%-2d          │\n", 
               i + 1, 
               teams[team_idx].batsmen[i].name,
               teams[team_idx].batsmen[i].stay_duration,
               teams[team_idx].batsmen[i].original_order + 1);
    }
    printf("  └─────┴──────────────────────────┴───────────────┴──────────────────┘\n\n");
}

void apply_batting_schedule(int team_idx, int sched_type) {
    match_state.batting_sched_type = sched_type;
    
    if (sched_type == SCHED_FCFS) {
        apply_fcfs_order(team_idx);
    } else if (sched_type == SCHED_SJF) {
        apply_sjf_order(team_idx);
    }
}

/*============================================================================
 * CONTEXT AND SCHEDULER FUNCTIONS
 *============================================================================*/

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
        
        // Can't bowl more than 4 overs, can't bowl consecutive overs
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
            
            // Death over specialists get highest priority in death overs
            if (match_state.current_over >= DEATH_OVER_START && bowlers[i].is_death_specialist) {
                effective_priority = 0;
            }
            // Powerplay specialists in powerplay
            else if (match_state.current_over < POWERPLAY_END && bowlers[i].is_powerplay_specialist) {
                effective_priority = 0;
            }
            
            // Consider economy rate
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
    
    char summary[512];
    snprintf(summary, sizeof(summary),
             YELLOW "\n╔══════════════════════════════════════════════════════════════════════╗\n"
             "║ END OF OVER %2d │ %-18s %d-%d-%d-%d %s\n"
             "║ Score: %s %d/%d │ Run Rate: %.2f │ Balls Remaining: %d\n"
             "╚══════════════════════════════════════════════════════════════════════╝" RESET,
             match_state.current_over + 1,
             bowlers[bowler].name,
             bowlers[bowler].overs_bowled + 1,
             bowlers[bowler].maidens,
             bowlers[bowler].runs_given,
             bowlers[bowler].wickets,
             maiden ? "(MAIDEN!)" : "",
             match_state.team_batting,
             match_state.total_runs,
             match_state.wickets,
             match_state.current_run_rate,
             match_state.balls_remaining - 6);
    
    printf("%s\n", summary);
    fprintf(match_log, "%s\n", summary);
    
    pthread_mutex_unlock(&log_mutex);
}

void context_switch_bowler(void) {
    pthread_mutex_lock(&scheduler_mutex);
    pthread_mutex_lock(&index_mutex);
    
    int old_bowler = current_bowler_idx;
    
    // Check for maiden
    bool maiden = (match_state.runs_in_current_over == 0);
    if (maiden && bowlers[old_bowler].balls_in_current_over >= 6) {
        bowlers[old_bowler].maidens++;
    }
    
    bowlers[old_bowler].overs_bowled++;
    
    // Calculate economy
    int total_balls = bowlers[old_bowler].overs_bowled * 6;
    if (total_balls > 0) {
        bowlers[old_bowler].economy = (float)bowlers[old_bowler].runs_given / 
                                       ((float)total_balls / 6.0);
    }
    
    // Print over summary
    pthread_mutex_unlock(&index_mutex);
    pthread_mutex_unlock(&scheduler_mutex);
    
    print_over_summary();
    
    pthread_mutex_lock(&scheduler_mutex);
    pthread_mutex_lock(&index_mutex);
    
    char log_msg[256];
    
    // Select next bowler based on match phase
    if (match_state.current_over >= DEATH_OVER_START - 1 || 
        match_state.match_intensity > HIGH_INTENSITY_THRESHOLD) {
        scheduler.scheduling_algo = 2;  // Priority
        current_bowler_idx = select_next_bowler_priority();
        snprintf(log_msg, sizeof(log_msg), 
                 MAGENTA "  [PRIORITY SCHEDULING] High intensity/Death overs phase" RESET);
    } else if (match_state.current_over < POWERPLAY_END) {
        scheduler.scheduling_algo = 2;  // Priority for powerplay
        current_bowler_idx = select_next_bowler_priority();
        snprintf(log_msg, sizeof(log_msg), 
                 CYAN "  [POWERPLAY] Specialist bowler selected" RESET);
    } else {
        scheduler.scheduling_algo = 0;  // Round Robin
        current_bowler_idx = select_next_bowler_rr();
        snprintf(log_msg, sizeof(log_msg), 
                 CYAN "  [ROUND ROBIN] Time Quantum = %d balls (1 over)" RESET, TIME_QUANTUM);
    }
    log_event(log_msg);
    
    scheduler.last_bowler = old_bowler;
    bowlers[current_bowler_idx].balls_in_current_over = 0;
    
    snprintf(log_msg, sizeof(log_msg), 
             GREEN "  New bowler: %s (Overs: %d/4, Wickets: %d, Econ: %.2f)" RESET,
             bowlers[current_bowler_idx].name,
             bowlers[current_bowler_idx].overs_bowled,
             bowlers[current_bowler_idx].wickets,
             bowlers[current_bowler_idx].economy);
    log_event(log_msg);
    
    // Switch strike at end of over
    int temp = striker_idx;
    striker_idx = non_striker_idx;
    non_striker_idx = temp;
    crease_ends[0].occupant_id = striker_idx;
    crease_ends[1].occupant_id = non_striker_idx;
    
    snprintf(log_msg, sizeof(log_msg), 
             "  Strike rotated: %s* (facing) | %s",
             batsmen[striker_idx].name, batsmen[non_striker_idx].name);
    log_event(log_msg);
    
    match_state.current_over++;
    match_state.current_ball = 0;
    match_state.runs_in_current_over = 0;
    scheduler.current_quantum_used = 0;
    
    pthread_mutex_unlock(&index_mutex);
    pthread_mutex_unlock(&scheduler_mutex);
}

/*============================================================================
 * DEADLOCK DETECTION (WAIT-FOR GRAPH)
 *============================================================================*/

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
    char log_msg[256];
    snprintf(log_msg, sizeof(log_msg), 
             RED "\n  ⚠ DEADLOCK DETECTED! Third Umpire (Resource Scheduler) reviewing..." RESET);
    log_event(log_msg);
    
    int batsman_to_dismiss = -1;
    
    pthread_mutex_lock(&waitgraph_mutex);
    
    // Find the batsman who is out (usually the one with fewer runs - less priority)
    for (int i = 0; i < NUM_BATSMEN; i++) {
        if (wait_graph.is_waiting[i]) {
            if (batsman_to_dismiss == -1 || 
                batsmen[i].runs_scored < batsmen[batsman_to_dismiss].runs_scored) {
                batsman_to_dismiss = i;
            }
        }
    }
    
    // Clear all waiting states
    for (int i = 0; i < NUM_BATSMEN; i++) {
        wait_graph.is_waiting[i] = false;
        wait_graph.waiting_for_resource[i] = -1;
        for (int j = 0; j < NUM_BATSMEN; j++) {
            wait_graph.adj_matrix[i][j] = 0;
        }
    }
    
    pthread_mutex_unlock(&waitgraph_mutex);
    
    if (batsman_to_dismiss != -1) {
        char dismissed_name[50];
        strncpy(dismissed_name, batsmen[batsman_to_dismiss].name, 49);
        dismissed_name[49] = '\0';
        
        batsmen[batsman_to_dismiss].is_out = true;
        snprintf(batsmen[batsman_to_dismiss].dismissal, 100, 
                 "run out (Third Umpire - Deadlock Resolution)");
        
        snprintf(log_msg, sizeof(log_msg), 
                 RED "  🔴 OUT! %s RUN OUT (Circular Wait Detected - Process Killed)" RESET,
                 dismissed_name);
        log_event(log_msg);
    }
    
    return batsman_to_dismiss;
}

void handle_runout_scenario(const BallContext* ctx) {
    int runs_attempted = pitch_ball.runs_scored;
    
    // Boundaries don't require running
    if (runs_attempted <= 0 || runs_attempted >= 4) return;
    
    pthread_mutex_lock(&index_mutex);
    int current_striker = striker_idx;
    int current_non_striker = non_striker_idx;
    pthread_mutex_unlock(&index_mutex);
    
    // Simulate running between wickets - potential circular wait
    for (int run = 0; run < runs_attempted; run++) {
        // Set up circular wait scenario
        // Batsman A wants End 2 (Resource 2), holds End 1 (Resource 1)
        // Batsman B wants End 1 (Resource 1), holds End 2 (Resource 2)
        add_edge_waitgraph(current_striker, current_non_striker);
        add_edge_waitgraph(current_non_striker, current_striker);
        
        // Check for deadlock with probability of run-out
        if (detect_deadlock()) {
            // 12% base probability, increases with match intensity
            int runout_prob = 12 + (match_state.match_intensity / 10);
            
            if (rand() % 100 < runout_prob) {
                int dismissed = resolve_deadlock();
                if (dismissed != -1) {
                    pthread_mutex_lock(&score_mutex);
                    match_state.wickets++;
                    pthread_mutex_unlock(&score_mutex);
                    
                    pitch_ball.is_wicket = true;
                    pitch_ball.wicket_type = 4;  // Run out
                    pitch_ball.dismissed_batsman_idx = dismissed;
                    strncpy(pitch_ball.dismissed_batsman_name, 
                            batsmen[dismissed].name, 49);
                    
                    new_batsman();
                    break;
                }
            }
        }
        
        // Clear edges on successful run
        remove_edge_waitgraph(current_striker, current_non_striker);
        remove_edge_waitgraph(current_non_striker, current_striker);
    }
    
    // Switch strike for odd runs
    if (runs_attempted % 2 == 1 && !pitch_ball.is_wicket) {
        pthread_mutex_lock(&index_mutex);
        int temp = striker_idx;
        striker_idx = non_striker_idx;
        non_striker_idx = temp;
        pthread_mutex_unlock(&index_mutex);
    }
}

/*============================================================================
 * MATCH MECHANICS
 *============================================================================*/

const char* generate_commentary(int outcome, const BallContext* ctx) {
    static char commentary[200];
    
    const char* four_comments[] = {
        "Beautiful shot! Races to the boundary!",
        "FOUR! Perfectly timed through the gap!",
        "Cracking drive! The ball flies to the fence!",
        "Elegant stroke! Four runs!",
        "Punched through covers for FOUR!"
    };
    
    const char* six_comments[] = {
        "MASSIVE SIX! Into the crowd!",
        "SIX! That's gone miles!",
        "HUGE HIT! Over the ropes!",
        "Maximum! Clean strike into the stands!",
        "SIX! What a shot! The crowd goes wild!"
    };
    
    const char* wicket_comments[] = {
        "GOT HIM! Big wicket!",
        "OUT! The bowler celebrates!",
        "WICKET! The stumps are shattered!",
        "That's the breakthrough!",
        "Clean bowled! What a delivery!"
    };
    
    const char* dot_comments[] = {
        "Defended solidly.",
        "Good length, no run.",
        "Played watchfully.",
        "Dot ball. Good bowling.",
        "Left alone outside off."
    };
    
    switch (outcome) {
        case OUTCOME_FOUR:
            strcpy(commentary, four_comments[rand() % 5]);
            break;
        case OUTCOME_SIX:
            strcpy(commentary, six_comments[rand() % 5]);
            break;
        case OUTCOME_WICKET:
            strcpy(commentary, wicket_comments[rand() % 5]);
            break;
        case OUTCOME_DOT:
            strcpy(commentary, dot_comments[rand() % 5]);
            break;
        case OUTCOME_SINGLE:
            strcpy(commentary, "Quick single taken.");
            break;
        case OUTCOME_DOUBLE:
            strcpy(commentary, "Good running! Two runs.");
            break;
        case OUTCOME_TRIPLE:
            strcpy(commentary, "Excellent running between wickets! Three!");
            break;
        case OUTCOME_WIDE:
            strcpy(commentary, "Wide ball signaled by the umpire.");
            break;
        case OUTCOME_NOBALL:
            strcpy(commentary, "No ball! Free hit coming up!");
            break;
        default:
            strcpy(commentary, "");
    }
    
    return commentary;
}

int generate_ball_outcome(const BallContext* ctx) {
    int rand_val = rand() % 1000;  // More granular randomization
    int striker = ctx->striker_idx;
    int bowler = ctx->bowler_idx;
    
    // Base skills
    int bat_skill = 700 - (batsmen[striker].is_tail_ender ? 250 : 0);
    int bowl_skill = 600 + (bowlers[bowler].is_death_specialist ? 80 : 0);
    
    // Match situation modifiers
    if (match_state.match_intensity > 70) bat_skill += 50;
    if (match_state.current_over >= DEATH_OVER_START) {
        bat_skill += 30;  // Batsmen take more risks
        bowl_skill += 20; // Bowlers under pressure
    }
    if (match_state.current_over < POWERPLAY_END) {
        bat_skill += 40;  // Powerplay aggression
    }
    
    // Chasing pressure in second innings
    if (!match_state.is_first_innings && match_state.target > 0) {
        int needed = match_state.target - match_state.total_runs;
        if (match_state.required_run_rate > 12.0) {
            bat_skill += 60;  // More aggressive
        }
        if (needed <= 20 && match_state.balls_remaining <= 12) {
            bat_skill += 80;  // Final push
        }
    }
    
    // Outcome probabilities (per 1000)
    if (rand_val < 25) return OUTCOME_WIDE;
    if (rand_val < 40) return OUTCOME_NOBALL;
    
    // Wicket probability based on skill differential
    int wicket_prob = 50 + (bowl_skill - bat_skill) / 8;
    if (wicket_prob < 30) wicket_prob = 30;
    if (wicket_prob > 120) wicket_prob = 120;
    
    if (rand_val < 40 + wicket_prob) return OUTCOME_WICKET;
    
    // Scoring shots
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
    strcpy(pitch_ball.commentary, generate_commentary(outcome, ctx));
    
    switch (outcome) {
        case OUTCOME_DOT:
            runs = 0;
            strcpy(pitch_ball.outcome_str, "•");
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
            
            const char* wkt_types[] = {"BOWLED!", "CAUGHT!", "LBW!", "STUMPED!", "C&B!"};
            strcpy(pitch_ball.outcome_str, wkt_types[wicket_type]);
            
            if (wicket_type == 1 || wicket_type == 4) ball_in_air = true;
            
            const char* dismissal_formats[] = {
                "b %s", "c sub b %s", "lbw b %s", "st †sub b %s", "c & b %s"
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
    
    // Update scores
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
    
    // Build description
    snprintf(pitch_ball.description, sizeof(pitch_ball.description),
             "%d.%d  %s to %s, %s",
             ctx->over_number, ctx->ball_number + 1,
             ctx->bowler_name, ctx->striker_name, pitch_ball.outcome_str);
    
    if (wicket) {
        match_state.wickets++;
        bowlers[bowler].wickets++;
    }
    
    // Update economy
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
    
    // FCFS: Simply get the next batsman in order
    // SJF: Already sorted, so next in order is shortest job
    int new_batsman_idx = next_batsman_idx;
    
    // Find next available batsman who is not out
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
        
        char log_msg[200];
        snprintf(log_msg, sizeof(log_msg), 
                 CYAN "  ▶ New batsman: %s comes to the crease" RESET,
                 batsmen[new_batsman_idx].name);
        log_event(log_msg);
        
        // Log scheduling info
        if (match_state.batting_sched_type == SCHED_FCFS) {
            snprintf(log_msg, sizeof(log_msg), 
                     "    [FCFS] Batting at original position #%d",
                     batsmen[new_batsman_idx].original_order + 1);
        } else {
            snprintf(log_msg, sizeof(log_msg), 
                     "    [SJF] Expected stay: %d balls (Original #%d)",
                     batsmen[new_batsman_idx].stay_duration,
                     batsmen[new_batsman_idx].original_order + 1);
        }
        log_event(log_msg);
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

/*============================================================================
 * THREAD FUNCTIONS
 *============================================================================*/

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
        
        usleep(120000);  // 120ms between balls
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
            char msg[100];
            snprintf(msg, sizeof(msg), 
                     YELLOW "\n══════════════════ ALL OUT! ══════════════════" RESET);
            log_event(msg);
            match_running = false;
        }
        
        if (match_state.current_over >= MAX_OVERS) {
            char msg[100];
            snprintf(msg, sizeof(msg), 
                     YELLOW "\n══════════════════ 20 OVERS COMPLETE! ══════════════════" RESET);
            log_event(msg);
            match_running = false;
        }
        
        if (!match_state.is_first_innings && match_state.total_runs >= match_state.target) {
            char msg[100];
            snprintf(msg, sizeof(msg), 
                     GREEN "\n══════════════════ TARGET ACHIEVED! ══════════════════" RESET);
            log_event(msg);
            match_running = false;
        }
        
        pthread_mutex_unlock(&match_mutex);
        
        usleep(80000);
    }
    return NULL;
}

void* scoreboard_thread_func(void* arg) {
    (void)arg;
    
    while (match_running) {
        pthread_mutex_lock(&score_mutex);
        pthread_mutex_lock(&index_mutex);
        
        int s_idx = striker_idx;
        int ns_idx = non_striker_idx;
        
        printf("\r" BOLD "[%s: %d/%d (%d.%d)]" RESET " %s %d(%d)* | %s %d(%d) | RR: %.2f",
               match_state.team_batting,
               match_state.total_runs,
               match_state.wickets,
               match_state.current_over,
               match_state.current_ball,
               batsmen[s_idx].name,
               batsmen[s_idx].runs_scored,
               batsmen[s_idx].balls_faced,
               batsmen[ns_idx].name,
               batsmen[ns_idx].runs_scored,
               batsmen[ns_idx].balls_faced,
               match_state.current_run_rate);
        
        if (!match_state.is_first_innings && match_state.target > 0) {
            int needed = match_state.target - match_state.total_runs;
            printf(" | Need %d off %d (RRR: %.2f)", 
                   needed, match_state.balls_remaining, match_state.required_run_rate);
        }
        
        printf("   ");
        fflush(stdout);
        
        pthread_mutex_unlock(&index_mutex);
        pthread_mutex_unlock(&score_mutex);
        
        sleep(1);
    }
    return NULL;
}

void* third_umpire_thread_func(void* arg) {
    (void)arg;
    
    while (match_running) {
        pthread_mutex_lock(&scheduler_mutex);
        
        if (match_state.current_over >= DEATH_OVER_START && scheduler.scheduling_algo != 2) {
            log_event(MAGENTA "  [Third Umpire] Recommending Priority Scheduling for death overs" RESET);
            scheduler.scheduling_algo = 2;
        }
        
        pthread_mutex_unlock(&scheduler_mutex);
        
        pthread_mutex_lock(&waitgraph_mutex);
        bool any_waiting = false;
        for (int i = 0; i < NUM_BATSMEN; i++) {
            if (wait_graph.is_waiting[i]) {
                any_waiting = true;
                break;
            }
        }
        
        if (any_waiting) {
            log_event(YELLOW "  [Third Umpire] Monitoring run-out situation (Circular Wait check)..." RESET);
            usleep(150000);
            if (detect_deadlock()) {
                resolve_deadlock();
            }
        }
        pthread_mutex_unlock(&waitgraph_mutex);
        
        usleep(200000);
    }
    return NULL;
}

/*============================================================================
 * OUTPUT FUNCTIONS
 *============================================================================*/

void log_event(const char* event) {
    pthread_mutex_lock(&log_mutex);
    fprintf(match_log, "%s\n", event);
    fflush(match_log);
    pthread_mutex_unlock(&log_mutex);
}

void print_ball_log(const BallContext* ctx) {
    pthread_mutex_lock(&log_mutex);
    
    printf("\n%s", pitch_ball.description);
    fprintf(match_log, "%s", pitch_ball.description);
    
    if (strlen(pitch_ball.commentary) > 0) {
        printf(" - %s", pitch_ball.commentary);
        fprintf(match_log, " - %s", pitch_ball.commentary);
    }
    printf("\n");
    fprintf(match_log, "\n");
    
    if (pitch_ball.is_wicket) {
        const char* dismissed_name = strlen(pitch_ball.dismissed_batsman_name) > 0 ? 
                                     pitch_ball.dismissed_batsman_name : ctx->striker_name;
        int dismissed_idx = pitch_ball.dismissed_batsman_idx >= 0 ? 
                            pitch_ball.dismissed_batsman_idx : ctx->striker_idx;
        
        printf(RED "  🔴 OUT! %s %d(%d) - %s" RESET "\n",
               dismissed_name,
               batsmen[dismissed_idx].runs_scored,
               batsmen[dismissed_idx].balls_faced,
               batsmen[dismissed_idx].dismissal);
        fprintf(match_log, "  OUT! %s %d(%d) - %s\n",
                dismissed_name,
                batsmen[dismissed_idx].runs_scored,
                batsmen[dismissed_idx].balls_faced,
                batsmen[dismissed_idx].dismissal);
    }
    
    if (pitch_ball.runs_scored == 4) {
        printf(GREEN "  🟢 FOUR!" RESET "\n");
    } else if (pitch_ball.runs_scored == 6) {
        printf(GREEN "  🟢 SIX!" RESET "\n");
    }
    
    fflush(stdout);
    pthread_mutex_unlock(&log_mutex);
}

void print_scorecard(void) {
    printf("\n");
    printf(CYAN "╔═════════════════════════════════════════════════════════════════════════════╗" RESET "\n");
    printf(CYAN "║                          SCORECARD - %-20s                      ║" RESET "\n", match_state.team_batting);
    printf(CYAN "║                          Scheduling: %-4s                                   ║" RESET "\n",
           match_state.batting_sched_type == SCHED_FCFS ? "FCFS" : "SJF");
    printf(CYAN "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf(CYAN "║ %-26s %-28s %4s %4s %3s %3s  SR   ║" RESET "\n", 
           "BATSMAN", "DISMISSAL", "R", "B", "4", "6");
    printf(CYAN "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    
    for (int i = 0; i < NUM_BATSMEN; i++) {
        if (batsmen[i].balls_faced > 0 || batsmen[i].is_out) {
            bool is_current = (i == striker_idx || i == non_striker_idx) && !batsmen[i].is_out;
            float sr = batsmen[i].balls_faced > 0 ? 
                       (float)batsmen[i].runs_scored * 100.0 / batsmen[i].balls_faced : 0.0;
            
            printf("║ %-26s %-28s %4d %4d %3d %3d %6.1f ║\n",
                   batsmen[i].name,
                   batsmen[i].is_out ? batsmen[i].dismissal : (is_current ? "not out *" : "not out"),
                   batsmen[i].runs_scored,
                   batsmen[i].balls_faced,
                   batsmen[i].fours,
                   batsmen[i].sixes,
                   sr);
        }
    }
    
    printf(CYAN "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf("║ Extras: (w %d, nb %d, lb %d) %50d ║\n", 
           match_state.wides, match_state.noballs, match_state.leg_byes, match_state.extras);
    printf(CYAN "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf("║ " GREEN "TOTAL: %d/%d (%d.%d overs)                              RR: %.2f" RESET "%13s ║\n",
           match_state.total_runs, match_state.wickets,
           match_state.current_over, match_state.current_ball,
           match_state.current_run_rate, "");
    printf(CYAN "╚═════════════════════════════════════════════════════════════════════════════╝" RESET "\n");
    
    // Bowling figures
    printf("\n");
    printf(YELLOW "╔═════════════════════════════════════════════════════════════════════════════╗" RESET "\n");
    printf(YELLOW "║                          BOWLING - %-20s                       ║" RESET "\n", match_state.team_bowling);
    printf(YELLOW "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf(YELLOW "║ %-26s %6s %6s %6s %6s %6s %10s ║" RESET "\n", 
           "BOWLER", "O", "M", "R", "W", "DOT", "ECON");
    printf(YELLOW "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    
    for (int i = 0; i < NUM_BOWLERS; i++) {
        int total_balls = bowlers[i].overs_bowled * 6 + bowlers[i].balls_in_current_over;
        if (total_balls > 0) {
            float economy = (float)bowlers[i].runs_given / ((float)total_balls / 6.0);
            printf("║ %-26s %4d.%d %6d %6d %6d %6d %10.2f ║\n",
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
    printf(YELLOW "╚═════════════════════════════════════════════════════════════════════════════╝" RESET "\n");
}

void print_innings_summary(int team_idx) {
    printf("\n");
    printf(MAGENTA "═══════════════════════════════════════════════════════════════════════════════\n" RESET);
    printf(MAGENTA "  INNINGS %d COMPLETE\n" RESET, team_idx + 1);
    printf(MAGENTA "  %s: %d/%d in %d.%d overs (Run Rate: %.2f)\n" RESET,
           teams[team_idx].name,
           teams[team_idx].innings_score,
           teams[team_idx].innings_wickets,
           teams[team_idx].innings_overs,
           teams[team_idx].innings_balls,
           teams[team_idx].innings_run_rate);
    printf(MAGENTA "  Batting Schedule Used: %s\n" RESET,
           match_state.batting_sched_type == SCHED_FCFS ? "FCFS (First Come First Serve)" : "SJF (Shortest Job First)");
    printf(MAGENTA "═══════════════════════════════════════════════════════════════════════════════\n" RESET);
}

void print_final_result(void) {
    printf("\n\n");
    printf(GREEN "╔═════════════════════════════════════════════════════════════════════════════╗" RESET "\n");
    printf(GREEN "║                            🏆 MATCH RESULT 🏆                               ║" RESET "\n");
    printf(GREEN "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf("║                                                                               ║\n");
    printf("║  %-15s: %3d/%d (%d.%d overs)   RR: %.2f%27s║\n",
           teams[0].name, teams[0].innings_score, teams[0].innings_wickets,
           teams[0].innings_overs, teams[0].innings_balls,
           teams[0].innings_run_rate, "");
    printf("║                                                                               ║\n");
    printf("║  %-15s: %3d/%d (%d.%d overs)   RR: %.2f%27s║\n",
           teams[1].name, teams[1].innings_score, teams[1].innings_wickets,
           teams[1].innings_overs, teams[1].innings_balls,
           teams[1].innings_run_rate, "");
    printf("║                                                                               ║\n");
    printf(GREEN "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    
    if (teams[1].innings_score > teams[0].innings_score) {
        int wickets_remaining = MAX_WICKETS - teams[1].innings_wickets;
        int balls_remaining = (MAX_OVERS * 6) - (teams[1].innings_overs * 6 + teams[1].innings_balls);
        printf(GREEN "║                                                                               ║\n");
        printf(GREEN "║     🎉 %s WINS by %d wickets (with %d balls remaining)!%15s║\n" RESET,
               teams[1].name, wickets_remaining, balls_remaining, "");
        printf(GREEN "║                                                                               ║\n");
    } else if (teams[0].innings_score > teams[1].innings_score) {
        int margin = teams[0].innings_score - teams[1].innings_score;
        printf(GREEN "║                                                                               ║\n");
        printf(GREEN "║     🎉 %s WINS by %d runs!%43s║\n" RESET,
               teams[0].name, margin, "");
        printf(GREEN "║                                                                               ║\n");
    } else {
        printf(YELLOW "║                                                                               ║\n");
        printf(YELLOW "║     🤝 MATCH TIED! Super Over required!%36s║\n" RESET, "");
        printf(YELLOW "║                                                                               ║\n");
    }
    
    printf(GREEN "╚═════════════════════════════════════════════════════════════════════════════╝" RESET "\n");
    
    // Log to file
    fprintf(match_log, "\n═══════════════════════════════════════════════════════════════════════════════\n");
    fprintf(match_log, "                                FINAL RESULT\n");
    fprintf(match_log, "═══════════════════════════════════════════════════════════════════════════════\n");
    fprintf(match_log, "%s: %d/%d (%d.%d overs)\n", 
            teams[0].name, teams[0].innings_score, teams[0].innings_wickets,
            teams[0].innings_overs, teams[0].innings_balls);
    fprintf(match_log, "%s: %d/%d (%d.%d overs)\n", 
            teams[1].name, teams[1].innings_score, teams[1].innings_wickets,
            teams[1].innings_overs, teams[1].innings_balls);
    
    if (teams[1].innings_score > teams[0].innings_score)
        fprintf(match_log, "\n%s WINS by %d wickets!\n", 
                teams[1].name, MAX_WICKETS - teams[1].innings_wickets);
    else if (teams[0].innings_score > teams[1].innings_score)
        fprintf(match_log, "\n%s WINS by %d runs!\n", 
                teams[0].name, teams[0].innings_score - teams[1].innings_score);
    else
        fprintf(match_log, "\nMATCH TIED!\n");
    
    fprintf(match_log, "═══════════════════════════════════════════════════════════════════════════════\n");
}

void print_match_summary(void) {
    printf("\n");
    printf(MAGENTA "╔═════════════════════════════════════════════════════════════════════════════╗" RESET "\n");
    printf(MAGENTA "║                          TECHNICAL SUMMARY                                  ║" RESET "\n");
    printf(MAGENTA "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf("║ " CYAN "T20 WORLD CUP 2026 SIMULATOR - IMPLEMENTATION DETAILS" RESET "%22s ║\n", "");
    printf(MAGENTA "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf("║ " YELLOW "THREADING MODEL (Players as Threads):" RESET "%38s ║\n", "");
    printf("║   • Bowler Threads: %d (1 active per delivery - writes to Pitch)%13s ║\n", NUM_BOWLERS, "");
    printf("║   • Batsman Threads: %d (2 active at crease - reads from Pitch)%12s ║\n", NUM_BATSMEN, "");
    printf("║   • Fielder Threads: %d (Sleep until ball_in_air signal)%19s ║\n", NUM_FIELDERS, "");
    printf("║   • Umpire Thread: 1 (Match state monitor + deadlock detection)%12s ║\n", "");
    printf("║   • Third Umpire: 1 (Resource Scheduler)%35s ║\n", "");
    printf("║   • Scoreboard Thread: 1 (Periodic display update)%25s ║\n", "");
    printf(MAGENTA "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf("║ " YELLOW "CRITICAL SECTION (The Pitch):" RESET "%47s ║\n", "");
    printf("║   • Protected by pitch_mutex - prevents race conditions%20s ║\n", "");
    printf("║   • Only one bowler writes (delivers ball) at a time%23s ║\n", "");
    printf("║   • Batsman must finish stroke before next ball%28s ║\n", "");
    printf(MAGENTA "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf("║ " YELLOW "SYNCHRONIZATION PRIMITIVES:" RESET "%49s ║\n", "");
    printf("║   • Mutexes: pitch, score, scheduler, index, waitgraph, log, match%8s ║\n", "");
    printf("║   • Semaphores: crease_sem (capacity=2), ball_ready, stroke_complete%7s ║\n", "");
    printf("║   • Condition Variables: ball_hit_cond (fielder wake signal)%15s ║\n", "");
    printf(MAGENTA "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf("║ " YELLOW "BATSMAN SCHEDULING (User Selected):" RESET "%41s ║\n", "");
    printf("║   • FCFS (First Come First Serve):%42s ║\n", "");
    printf("║       - Batsmen bat in original designated order (1 to 11)%17s ║\n", "");
    printf("║       - No reordering - arrival order determines sequence%19s ║\n", "");
    printf("║   • SJF (Shortest Job First):%47s ║\n", "");
    printf("║       - Sorted by expected stay duration (ascending)%23s ║\n", "");
    printf("║       - Tail-enders bat before middle order%33s ║\n", "");
    printf(MAGENTA "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf("║ " YELLOW "BOWLER SCHEDULING:" RESET "%58s ║\n", "");
    printf("║   • Round Robin: Time Quantum = %d balls (1 over)%28s ║\n", TIME_QUANTUM, "");
    printf("║   • Priority: Death specialists in overs %d-%d%31s ║\n", DEATH_OVER_START, MAX_OVERS, "");
    printf("║   • Context Switch: Save/restore bowler state between overs%16s ║\n", "");
    printf(MAGENTA "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf("║ " YELLOW "DEADLOCK HANDLING (Run-out Scenario):" RESET "%39s ║\n", "");
    printf("║   • Circular Wait: Both batsmen wanting each other's crease end%12s ║\n", "");
    printf("║   • Detection: Wait-for Graph with DFS cycle detection%21s ║\n", "");
    printf("║   • Resolution: Third Umpire kills one process (batsman OUT)%15s ║\n", "");
    printf(MAGENTA "╚═════════════════════════════════════════════════════════════════════════════╝" RESET "\n");
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
    
    if (match_log) fclose(match_log);
}

/*============================================================================
 * INNINGS SIMULATION
 *============================================================================*/

int get_scheduling_choice(int innings_num) {
    int choice;
    int bat_team = (innings_num == 1) ? 0 : 1;
    
    printf("\n");
    printf(YELLOW "╔═════════════════════════════════════════════════════════════════════════════╗" RESET "\n");
    printf(YELLOW "║               SELECT BATSMAN SCHEDULING ALGORITHM                           ║" RESET "\n");
    printf(YELLOW "║                     INNINGS %d - %s BATTING                                ║" RESET "\n",
           innings_num, teams[bat_team].name);
    printf(YELLOW "╠═════════════════════════════════════════════════════════════════════════════╣" RESET "\n");
    printf("║                                                                               ║\n");
    printf("║  [1] " CYAN "FCFS - First Come First Serve" RESET "%45s ║\n", "");
    printf("║      • Batsmen bat in their original designated order (1 to 11)%12s ║\n", "");
    printf("║      • No reordering - natural arrival sequence%29s ║\n", "");
    printf("║      • Traditional batting order: Openers → Middle → Tail%19s ║\n", "");
    printf("║      • Best for: Stable strategy, predictable batting%22s ║\n", "");
    printf("║                                                                               ║\n");
    printf("║  [2] " CYAN "SJF - Shortest Job First" RESET "%51s ║\n", "");
    printf("║      • Batsmen sorted by expected stay duration (ascending)%16s ║\n", "");
    printf("║      • Tail-enders (shorter stay) bat before middle order%19s ║\n", "");
    printf("║      • Openers remain unchanged (positions 1-2)%29s ║\n", "");
    printf("║      • Best for: Quick turnover, experimental strategy%22s ║\n", "");
    printf("║                                                                               ║\n");
    printf(YELLOW "╚═════════════════════════════════════════════════════════════════════════════╝" RESET "\n");
    printf("\nEnter your choice (1 for FCFS, 2 for SJF): ");
    
    if (scanf("%d", &choice) != 1 || (choice != 1 && choice != 2)) {
        printf(YELLOW "Invalid choice. Defaulting to FCFS.\n" RESET);
        choice = 1;
        while (getchar() != '\n');
    }
    
    return choice;
}

void simulate_innings(int innings_num) {
    int bat_team = (innings_num == 1) ? 0 : 1;
    int bowl_team = (innings_num == 1) ? 1 : 0;
    
    // Get scheduling choice
    int sched_choice = get_scheduling_choice(innings_num);
    
    printf("\n");
    printf(CYAN "═══════════════════════════════════════════════════════════════════════════════\n" RESET);
    printf(CYAN "                              INNINGS %d\n" RESET, innings_num);
    printf(CYAN "                         %s BATTING\n" RESET, teams[bat_team].name);
    printf(CYAN "                         %s BOWLING\n" RESET, teams[bowl_team].name);
    printf(CYAN "                         Scheduling: %s\n" RESET, 
           sched_choice == SCHED_FCFS ? "FCFS (First Come First Serve)" : "SJF (Shortest Job First)");
    if (innings_num == 2) {
        printf(CYAN "                         TARGET: %d runs from %d overs\n" RESET, 
               match_state.target, MAX_OVERS);
    }
    printf(CYAN "═══════════════════════════════════════════════════════════════════════════════\n" RESET);
    
    // Reset for innings
    reset_for_innings(bat_team, bowl_team);
    
    // Apply batting schedule
    apply_batting_schedule(bat_team, sched_choice);
    
    match_state.is_first_innings = (innings_num == 1);
    
    fprintf(match_log, "\n\n═══════════════════════════════════════════════════════════════════════════════\n");
    fprintf(match_log, "                                 INNINGS %d\n", innings_num);
    fprintf(match_log, "                          %s vs %s\n", teams[bat_team].name, teams[bowl_team].name);
    fprintf(match_log, "                          Scheduling: %s\n", 
            sched_choice == SCHED_FCFS ? "FCFS" : "SJF");
    if (innings_num == 2) {
        fprintf(match_log, "                          Target: %d\n", match_state.target);
    }
    fprintf(match_log, "═══════════════════════════════════════════════════════════════════════════════\n\n");
    
    printf("\nPress ENTER to start innings %d...", innings_num);
    while (getchar() != '\n');
    getchar();
    printf("\n");
    
    // Create threads
    pthread_t bowler_threads[NUM_BOWLERS];
    pthread_t batsman_threads[NUM_BATSMEN];
    pthread_t fielder_threads[NUM_FIELDERS];
    pthread_t umpire_thread, scoreboard_thread, third_umpire_thread;
    
    int bowler_ids[NUM_BOWLERS];
    int batsman_ids[NUM_BATSMEN];
    int fielder_ids[NUM_FIELDERS];
    
    pthread_create(&umpire_thread, NULL, umpire_thread_func, NULL);
    pthread_create(&third_umpire_thread, NULL, third_umpire_thread_func, NULL);
    pthread_create(&scoreboard_thread, NULL, scoreboard_thread_func, NULL);
    
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
    
    // Wait for innings completion
    pthread_join(umpire_thread, NULL);
    
    // Stop threads
    match_running = false;
    
    for (int i = 0; i < 5; i++) {
        sem_post(&ball_ready_sem);
        sem_post(&stroke_complete_sem);
    }
    pthread_cond_broadcast(&ball_hit_cond);
    
    for (int i = 0; i < NUM_BOWLERS; i++) pthread_cancel(bowler_threads[i]);
    for (int i = 0; i < NUM_BATSMEN; i++) pthread_cancel(batsman_threads[i]);
    for (int i = 0; i < NUM_FIELDERS; i++) pthread_cancel(fielder_threads[i]);
    pthread_cancel(scoreboard_thread);
    pthread_cancel(third_umpire_thread);
    
    usleep(100000);
    
    // Save innings stats
    int total_balls = match_state.current_over * 6 + match_state.current_ball;
    teams[bat_team].innings_score = match_state.total_runs;
    teams[bat_team].innings_wickets = match_state.wickets;
    teams[bat_team].innings_overs = match_state.current_over;
    teams[bat_team].innings_balls = match_state.current_ball;
    teams[bat_team].innings_extras = match_state.extras;
    teams[bat_team].innings_run_rate = total_balls > 0 ? 
                                        (float)match_state.total_runs * 6.0 / total_balls : 0.0;
    
    // Print scorecard
    printf("\n\n");
    print_scorecard();
    print_innings_summary(bat_team);
    
    // Set target for second innings
    if (innings_num == 1) {
        match_state.target = teams[bat_team].innings_score + 1;
    }
}

/*============================================================================
 * MAIN FUNCTION
 *============================================================================*/

int main(void) {
    printf("\n");
    printf(CYAN "╔═════════════════════════════════════════════════════════════════════════════╗" RESET "\n");
    printf(CYAN "║                                                                             ║" RESET "\n");
    printf(CYAN "║          🏏  T20 CRICKET WORLD CUP 2026 SIMULATOR  🏏                      ║" RESET "\n");
    printf(CYAN "║                                                                             ║" RESET "\n");
    printf(CYAN "║                       Multi-threaded Full Match                             ║" RESET "\n");
    printf(CYAN "║                         20 Overs Per Innings                                ║" RESET "\n");
    printf(CYAN "║                       Batsman Scheduling: FCFS vs SJF                       ║" RESET "\n");
    printf(CYAN "║                                                                             ║" RESET "\n");
    printf(CYAN "╚═════════════════════════════════════════════════════════════════════════════╝" RESET "\n\n");
    
    printf(YELLOW "═══════════════════════════════════════════════════════════════════════════════\n" RESET);
    printf(YELLOW "                                 ASSUMPTIONS\n" RESET);
    printf(YELLOW "═══════════════════════════════════════════════════════════════════════════════\n" RESET);
    printf("  1.  T20 Format: %d overs per innings, %d wickets per side\n", MAX_OVERS, MAX_WICKETS);
    printf("  2.  Powerplay: Overs 1-%d (fielding restrictions apply)\n", POWERPLAY_END);
    printf("  3.  Death Overs: Overs %d-%d (specialist bowlers prioritized)\n", DEATH_OVER_START, MAX_OVERS);
    printf("  4.  Each bowler: Maximum 4 overs, cannot bowl consecutive overs\n");
    printf("  5.  Pitch is Critical Section: Protected by mutex (no race conditions)\n");
    printf("  6.  Crease has capacity 2: Enforced by semaphore (max 2 batsmen)\n");
    printf("  7.  Fielders sleep until ball_in_air: Condition variable signaling\n");
    printf("  8.  Run-out deadlock: Circular wait detected via Wait-for Graph + DFS\n");
    printf("  9.  Batsman scheduling: User selects FCFS or SJF per innings\n");
    printf("  10. Bowler scheduling: Round Robin (TQ=6) → Priority (death overs)\n");
    printf(YELLOW "═══════════════════════════════════════════════════════════════════════════════\n\n" RESET);
    
    srand(time(NULL));
    
    match_log = fopen("match_log.txt", "w");
    if (!match_log) {
        perror("Failed to open log file");
        return 1;
    }
    
    fprintf(match_log, "═══════════════════════════════════════════════════════════════════════════════\n");
    fprintf(match_log, "                T20 CRICKET WORLD CUP 2026 - BALL BY BALL LOG\n");
    fprintf(match_log, "═══════════════════════════════════════════════════════════════════════════════\n");
    time_t now = time(NULL);
    fprintf(match_log, "Generated: %s", ctime(&now));
    fprintf(match_log, "Format: T20 International (%d overs per innings)\n", MAX_OVERS);
    fprintf(match_log, "Batsman Scheduling Options: FCFS (First Come First Serve) / SJF (Shortest Job First)\n");
    fprintf(match_log, "═══════════════════════════════════════════════════════════════════════════════\n\n");
    
    printf(YELLOW "Initializing simulation...\n" RESET);
    initialize_teams();
    initialize_match_state();
    initialize_scheduler();
    initialize_synchronization();
    initialize_wait_graph();
    printf(GREEN "✓ All systems initialized successfully!\n\n" RESET);
    
    // ═══════════════════ FIRST INNINGS ═══════════════════
    simulate_innings(1);
    
    printf("\n");
    printf(YELLOW "═══════════════════════════════════════════════════════════════════════════════\n" RESET);
    printf(YELLOW "                               INNINGS BREAK\n" RESET);
    printf(YELLOW "                      %s need %d runs to win\n" RESET, 
           teams[1].name, match_state.target);
    printf(YELLOW "═══════════════════════════════════════════════════════════════════════════════\n" RESET);
    
    // ═══════════════════ SECOND INNINGS ═══════════════════
    simulate_innings(2);
    
    // ═══════════════════ FINAL RESULT ═══════════════════
    print_final_result();
    print_match_summary();
    
    cleanup_resources();
    
    printf(GREEN "\n✓ Match simulation completed successfully!\n" RESET);
    printf("  Detailed ball-by-ball log saved to: " CYAN "match_log.txt" RESET "\n\n");
    
    return 0;
}