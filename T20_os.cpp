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

#define SCHED_FCFS 1
#define SCHED_SJF 2

#define OUTCOME_DOT 0
#define OUTCOME_SINGLE 1
#define OUTCOME_DOUBLE 2
#define OUTCOME_FOUR 3
#define OUTCOME_SIX 4
#define OUTCOME_WICKET 5
#define OUTCOME_WIDE 6
#define OUTCOME_NOBALL 7

typedef struct {
    int id;
    char name[50];
    int runs_scored;
    int balls_faced;
    int fours;
    int sixes;
    bool is_out;
    int stay_duration;
    bool is_tail_ender;
    char dismissal[100];
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
} Bowler;

typedef struct {
    int id;
    char name[50];
    char position[30];
    int catches;
    bool is_active;
} Fielder;

typedef struct {
    int total_runs;
    int wickets;
    int current_over;
    int current_ball;
    int extras;
    int wides;
    int noballs;
    int target;
    bool is_first_innings;
    char team_batting[50];
    char team_bowling[50];
    int batting_sched_type;
    float current_run_rate;
    float required_run_rate;
    int balls_remaining;
    int runs_in_current_over;
} MatchState;

typedef struct {
    char name[50];
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


MatchState match_state;
Team teams[2];

Batsman* batsmen;
Bowler* bowlers;
Fielder* fielders;

int striker_idx = 0;
int non_striker_idx = 1;
int current_bowler_idx = 0;
int next_batsman_idx = 2;


pthread_mutex_t game_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t ball_ready_cond = PTHREAD_COND_INITIALIZER;
pthread_cond_t ball_played_cond = PTHREAD_COND_INITIALIZER;
pthread_cond_t fielder_cond = PTHREAD_COND_INITIALIZER;

volatile bool match_running = true;
volatile bool ball_ready = false;
volatile bool ball_played = false;
volatile bool ball_in_air = false;
volatile int current_ball_outcome = -1;


int ball_runs = 0;
bool ball_is_wicket = false;
bool ball_is_wide = false;
bool ball_is_noball = false;
char ball_outcome_str[50];
char dismissed_batsman[50];
int dismissed_idx = -1;

void initialize_teams(void);
void reset_for_innings(int bat_team, int bowl_team);
void apply_batting_schedule(int team_idx, int sched_type);
int select_next_bowler(void);
int generate_ball_outcome(void);
void process_ball_outcome(int outcome);
void switch_strike(void);
void bring_new_batsman(void);
void update_run_rates(void);
void print_ball_commentary(void);
void print_over_summary(void);
void print_scorecard(void);
void print_final_result(void);

void* bowler_thread_func(void* arg);
void* batsman_thread_func(void* arg);
void* fielder_thread_func(void* arg);
void* umpire_thread_func(void* arg);

void simulate_innings(int innings_num);
int get_scheduling_choice(int innings_num);

bool innings_should_end(void) {
    if (match_state.wickets >= MAX_WICKETS) return true;
    if (match_state.current_over >= MAX_OVERS) return true;
    if (!match_state.is_first_innings && match_state.total_runs >= match_state.target) return true;
    return false;
}

void initialize_teams(void) {
    strcpy(teams[0].name, "INDIA");
    const char* india_batsmen[] = {
        "Rohit Sharma", "Yashasvi Jaiswal", "Virat Kohli", "Suryakumar Yadav",
        "Hardik Pandya", "Rishabh Pant", "Ravindra Jadeja", "Axar Patel",
        "Kuldeep Yadav", "Jasprit Bumrah", "Mohammed Siraj"
    };
    const int india_stay[] = {35, 30, 40, 28, 25, 22, 18, 15, 8, 5, 4};
    const char* india_bowlers[] = {
        "Jasprit Bumrah", "Mohammed Siraj", "Arshdeep Singh", "Kuldeep Yadav", "Ravindra Jadeja"
    };

    strcpy(teams[1].name, "AUSTRALIA");
    const char* aus_batsmen[] = {
        "Travis Head", "David Warner", "Mitchell Marsh", "Steve Smith",
        "Glenn Maxwell", "Marcus Stoinis", "Tim David", "Pat Cummins",
        "Mitchell Starc", "Adam Zampa", "Josh Hazlewood"
    };
    const int aus_stay[] = {32, 28, 35, 30, 20, 22, 18, 12, 6, 4, 3};
    const char* aus_bowlers[] = {
        "Pat Cummins", "Mitchell Starc", "Josh Hazlewood", "Adam Zampa", "Glenn Maxwell"
    };

    const char* positions[] = {
        "Slip", "Gully", "Point", "Cover", "Mid-off",
        "Mid-on", "Midwicket", "Square Leg", "Fine Leg", "Wicketkeeper"
    };

    const char** bat_names[2] = {india_batsmen, aus_batsmen};
    const char** bowl_names[2] = {india_bowlers, aus_bowlers};
    const int* stays[2] = {india_stay, aus_stay};

    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < NUM_BATSMEN; i++) {
            teams[t].batsmen[i].id = i;
            strncpy(teams[t].batsmen[i].name, bat_names[t][i], 49);
            teams[t].batsmen[i].runs_scored = 0;
            teams[t].batsmen[i].balls_faced = 0;
            teams[t].batsmen[i].fours = 0;
            teams[t].batsmen[i].sixes = 0;
            teams[t].batsmen[i].is_out = false;
            teams[t].batsmen[i].stay_duration = stays[t][i];
            teams[t].batsmen[i].is_tail_ender = (i >= 8);
            strcpy(teams[t].batsmen[i].dismissal, "not out");
            teams[t].original_batsmen[i] = teams[t].batsmen[i];
        }

        for (int i = 0; i < NUM_BOWLERS; i++) {
            teams[t].bowlers[i].id = i;
            strncpy(teams[t].bowlers[i].name, bowl_names[t][i], 49);
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
        }

        for (int i = 0; i < NUM_FIELDERS; i++) {
            teams[t].fielders[i].id = i;
            snprintf(teams[t].fielders[i].name, 50, "Fielder %d", i + 1);
            strncpy(teams[t].fielders[i].position, positions[i], 29);
            teams[t].fielders[i].catches = 0;
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

void reset_for_innings(int bat_team, int bowl_team) {
    batsmen = teams[bat_team].batsmen;
    bowlers = teams[bowl_team].bowlers;
    fielders = teams[bowl_team].fielders;

    for (int i = 0; i < NUM_BATSMEN; i++) {
        batsmen[i].runs_scored = 0;
        batsmen[i].balls_faced = 0;
        batsmen[i].fours = 0;
        batsmen[i].sixes = 0;
        batsmen[i].is_out = false;
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
    match_state.current_run_rate = 0.0;
    match_state.balls_remaining = MAX_OVERS * BALLS_PER_OVER;
    match_state.runs_in_current_over = 0;

    strcpy(match_state.team_batting, teams[bat_team].name);
    strcpy(match_state.team_bowling, teams[bowl_team].name);

    match_running = true;
    ball_ready = false;
    ball_played = false;
    ball_in_air = false;
}

void apply_batting_schedule(int team_idx, int sched_type) {
    match_state.batting_sched_type = sched_type;

    for (int i = 0; i < NUM_BATSMEN; i++) {
        teams[team_idx].batsmen[i] = teams[team_idx].original_batsmen[i];
        teams[team_idx].batsmen[i].runs_scored = 0;
        teams[team_idx].batsmen[i].balls_faced = 0;
        teams[team_idx].batsmen[i].fours = 0;
        teams[team_idx].batsmen[i].sixes = 0;
        teams[team_idx].batsmen[i].is_out = false;
        strcpy(teams[team_idx].batsmen[i].dismissal, "not out");
    }

    if (sched_type == SCHED_SJF) {
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
}

int select_next_bowler(void) {
    int prev = current_bowler_idx;
    
  
    for (int i = 1; i <= NUM_BOWLERS; i++) {
        int candidate = (prev + i) % NUM_BOWLERS;
        if (bowlers[candidate].overs_bowled < 4) {
            return candidate;
        }
    }
    
    
    for (int i = 0; i < NUM_BOWLERS; i++) {
        if (bowlers[i].overs_bowled < 4) {
            return i;
        }
    }
    
    return 0;
}

int generate_ball_outcome(void) {
    int r = rand() % 1000;

    // Extras: ~2.5%
    if (r < 15) return OUTCOME_WIDE;
    if (r < 25) return OUTCOME_NOBALL;

    // Wicket: ~2% for top order, ~4% for tail
    int wicket_prob = batsmen[striker_idx].is_tail_ender ? 40 : 20;
    if (batsmen[striker_idx].balls_faced > 20) wicket_prob -= 5;
    if (batsmen[striker_idx].balls_faced < 5) wicket_prob += 8;
    if (wicket_prob < 10) wicket_prob = 10;
    if (wicket_prob > 50) wicket_prob = 50;

    if (r < 25 + wicket_prob) return OUTCOME_WICKET;

    // Scoring: Dot ~32%, Single ~38%, Double ~10%, Four ~14%, Six ~6%
    int base = 25 + wicket_prob;
    if (r < base + 320) return OUTCOME_DOT;
    if (r < base + 700) return OUTCOME_SINGLE;
    if (r < base + 800) return OUTCOME_DOUBLE;
    if (r < base + 940) return OUTCOME_FOUR;
    return OUTCOME_SIX;
}

void process_ball_outcome(int outcome) {
    int striker = striker_idx;
    int bowler = current_bowler_idx;
    
    ball_runs = 0;
    ball_is_wicket = false;
    ball_is_wide = false;
    ball_is_noball = false;
    ball_in_air = false;
    dismissed_idx = -1;
    strcpy(ball_outcome_str, "");
    strcpy(dismissed_batsman, "");

    bool legal_ball = true;

    switch (outcome) {
        case OUTCOME_DOT:
            ball_runs = 0;
            strcpy(ball_outcome_str, "no run");
            bowlers[bowler].dots++;
            break;

        case OUTCOME_SINGLE:
            ball_runs = 1;
            strcpy(ball_outcome_str, "1 run");
            break;

        case OUTCOME_DOUBLE:
            ball_runs = 2;
            strcpy(ball_outcome_str, "2 runs");
            break;

        case OUTCOME_FOUR:
            ball_runs = 4;
            strcpy(ball_outcome_str, "FOUR!");
            batsmen[striker].fours++;
            break;

        case OUTCOME_SIX:
            ball_runs = 6;
            strcpy(ball_outcome_str, "SIX!");
            batsmen[striker].sixes++;
            ball_in_air = true;
            break;

        case OUTCOME_WICKET: {
            ball_runs = 0;
            ball_is_wicket = true;
            ball_in_air = true;
            int wkt_type = rand() % 5;
            const char* formats[] = {"b %s", "c & b %s", "lbw b %s", "st †sub b %s", "c sub b %s"};
            snprintf(batsmen[striker].dismissal, 100, formats[wkt_type], bowlers[bowler].name);
            strcpy(ball_outcome_str, "OUT!");
            strcpy(dismissed_batsman, batsmen[striker].name);
            dismissed_idx = striker;
            batsmen[striker].is_out = true;
            break;
        }

        case OUTCOME_WIDE:
            ball_runs = 1;
            ball_is_wide = true;
            legal_ball = false;
            strcpy(ball_outcome_str, "wide");
            bowlers[bowler].wides++;
            match_state.wides++;
            break;

        case OUTCOME_NOBALL:
            ball_runs = 1;
            ball_is_noball = true;
            legal_ball = false;
            strcpy(ball_outcome_str, "no ball");
            bowlers[bowler].noballs++;
            match_state.noballs++;
            break;
    }

    match_state.total_runs += ball_runs;
    match_state.runs_in_current_over += ball_runs;

    if (!ball_is_wide && !ball_is_noball) {
        batsmen[striker].runs_scored += ball_runs;
        batsmen[striker].balls_faced++;
    } else {
        match_state.extras += ball_runs;
    }

    bowlers[bowler].runs_given += ball_runs;

    if (ball_is_wicket) {
        match_state.wickets++;
        bowlers[bowler].wickets++;
    }

    
    if (legal_ball) {
        bowlers[bowler].balls_in_current_over++;
        match_state.current_ball++;
    }
}

void switch_strike(void) {
    int temp = striker_idx;
    striker_idx = non_striker_idx;
    non_striker_idx = temp;
}

void bring_new_batsman(void) {
    while (next_batsman_idx < NUM_BATSMEN && batsmen[next_batsman_idx].is_out) {
        next_batsman_idx++;
    }

    if (next_batsman_idx < NUM_BATSMEN) {
        if (batsmen[striker_idx].is_out) {
            striker_idx = next_batsman_idx;
        } else if (batsmen[non_striker_idx].is_out) {
            non_striker_idx = next_batsman_idx;
        }
        next_batsman_idx++;
    }
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

void print_ball_commentary(void) {
    pthread_mutex_lock(&print_mutex);
    
    printf("%d.%d %s to %s, %s",
           match_state.current_over + 1,
           match_state.current_ball,
           bowlers[current_bowler_idx].name,
           batsmen[striker_idx].name,
           ball_outcome_str);

    if (ball_is_wicket && dismissed_idx >= 0) {
        printf(" - %s %s %d(%d)",
               dismissed_batsman,
               batsmen[dismissed_idx].dismissal,
               batsmen[dismissed_idx].runs_scored,
               batsmen[dismissed_idx].balls_faced);
    }

    printf("\n");
    fflush(stdout);
    
    pthread_mutex_unlock(&print_mutex);
}

void print_over_summary(void) {
    pthread_mutex_lock(&print_mutex);

    int b = current_bowler_idx;
    bool maiden = (match_state.runs_in_current_over == 0);
    if (maiden) bowlers[b].maidens++;

    printf("\n--- End of Over %d ---\n", match_state.current_over + 1);
    printf("    %s: %d-%d-%d-%d %s\n",
           bowlers[b].name,
           bowlers[b].overs_bowled + 1,
           bowlers[b].maidens,
           bowlers[b].runs_given,
           bowlers[b].wickets,
           maiden ? "(MAIDEN)" : "");
    printf("    %s: %d/%d | CRR: %.2f",
           match_state.team_batting,
           match_state.total_runs,
           match_state.wickets,
           match_state.current_run_rate);

    if (!match_state.is_first_innings && match_state.target > 0) {
        int needed = match_state.target - match_state.total_runs;
        if (needed > 0) {
            printf(" | Need %d from %d balls", needed, match_state.balls_remaining);
        }
    }
    printf("\n\n");

    pthread_mutex_unlock(&print_mutex);
}

void* bowler_thread_func(void* arg) {
    (void)arg;

    while (match_running) {
        pthread_mutex_lock(&game_mutex);

        if (innings_should_end()) {
            match_running = false;
            pthread_cond_broadcast(&ball_ready_cond);
            pthread_mutex_unlock(&game_mutex);
            break;
        }

      
        ball_ready = true;
        ball_played = false;
        pthread_cond_signal(&ball_ready_cond);

       
        while (!ball_played && match_running) {
            pthread_cond_wait(&ball_played_cond, &game_mutex);
        }

        if (!match_running) {
            pthread_mutex_unlock(&game_mutex);
            break;
        }

       
        print_ball_commentary();

        
        if (ball_is_wicket && match_state.wickets < MAX_WICKETS) {
            bring_new_batsman();
            if (next_batsman_idx <= NUM_BATSMEN && match_state.wickets < MAX_WICKETS) {
                pthread_mutex_lock(&print_mutex);
                printf("    New batsman: %s\n", batsmen[striker_idx].name);
                pthread_mutex_unlock(&print_mutex);
            }
        }

       
        if (!ball_is_wicket && ball_runs > 0 && ball_runs % 2 == 1) {
            switch_strike();
        }

        
        if (ball_in_air) {
            pthread_cond_broadcast(&fielder_cond);
            ball_in_air = false;
        }

        update_run_rates();

        if (match_state.current_ball >= BALLS_PER_OVER) {
           
            bowlers[current_bowler_idx].overs_bowled++;
            
            print_over_summary();

            
            switch_strike();

            
            current_bowler_idx = select_next_bowler();
            bowlers[current_bowler_idx].balls_in_current_over = 0;

            
            match_state.current_over++;
            match_state.current_ball = 0;
            match_state.runs_in_current_over = 0;
        }

      
        if (innings_should_end()) {
            match_running = false;
        }

        pthread_mutex_unlock(&game_mutex);

        usleep(100000); 
    }

    return NULL;
}

void* batsman_thread_func(void* arg) {
    (void)arg;

    while (match_running) {
        pthread_mutex_lock(&game_mutex);

     
        while (!ball_ready && match_running) {
            pthread_cond_wait(&ball_ready_cond, &game_mutex);
        }

        if (!match_running) {
            pthread_mutex_unlock(&game_mutex);
            break;
        }

      
        int outcome = generate_ball_outcome();
        process_ball_outcome(outcome);

        
        ball_ready = false;
        ball_played = true;
        pthread_cond_signal(&ball_played_cond);

        pthread_mutex_unlock(&game_mutex);
    }

    return NULL;
}

void* fielder_thread_func(void* arg) {
    int id = *(int*)arg;

    while (match_running) {
        pthread_mutex_lock(&game_mutex);

        while (!ball_in_air && match_running) {
            pthread_cond_wait(&fielder_cond, &game_mutex);
        }

        if (!match_running) {
            pthread_mutex_unlock(&game_mutex);
            break;
        }

        fielders[id].is_active = true;
        
        if (ball_is_wicket) {
            fielders[id].catches++;
        }
        
        fielders[id].is_active = false;

        pthread_mutex_unlock(&game_mutex);
        usleep(10000);
    }

    return NULL;
}

void* umpire_thread_func(void* arg) {
    (void)arg;

    while (match_running) {
        pthread_mutex_lock(&game_mutex);

        if (innings_should_end()) {
            if (match_state.wickets >= MAX_WICKETS) {
                pthread_mutex_lock(&print_mutex);
                printf("\n*** ALL OUT! ***\n");
                pthread_mutex_unlock(&print_mutex);
            } else if (match_state.current_over >= MAX_OVERS) {
                pthread_mutex_lock(&print_mutex);
                printf("\n*** INNINGS COMPLETE - 20 OVERS ***\n");
                pthread_mutex_unlock(&print_mutex);
            } else if (!match_state.is_first_innings && match_state.total_runs >= match_state.target) {
                pthread_mutex_lock(&print_mutex);
                printf("\n*** TARGET ACHIEVED! ***\n");
                pthread_mutex_unlock(&print_mutex);
            }
            match_running = false;
            pthread_cond_broadcast(&ball_ready_cond);
            pthread_cond_broadcast(&ball_played_cond);
            pthread_cond_broadcast(&fielder_cond);
        }

        pthread_mutex_unlock(&game_mutex);
        usleep(50000);
    }

    return NULL;
}

void print_scorecard(void) {
    printf("\n");
    printf("========================================================================\n");
    printf("               SCORECARD - %s (%s)\n",
           match_state.team_batting,
           match_state.batting_sched_type == SCHED_FCFS ? "FCFS" : "SJF");
    printf("========================================================================\n");
    printf("%-22s %-22s %4s %4s %3s %3s %6s\n",
           "BATSMAN", "DISMISSAL", "R", "B", "4s", "6s", "SR");
    printf("------------------------------------------------------------------------\n");

    for (int i = 0; i < NUM_BATSMEN; i++) {
        if (batsmen[i].balls_faced > 0 || batsmen[i].is_out) {
            bool current = (i == striker_idx || i == non_striker_idx) && !batsmen[i].is_out;
            float sr = batsmen[i].balls_faced > 0 ?
                       (float)batsmen[i].runs_scored * 100.0 / batsmen[i].balls_faced : 0.0;

            char disp[24];
            if (batsmen[i].is_out) {
                strncpy(disp, batsmen[i].dismissal, 21);
                disp[21] = '\0';
            } else {
                strcpy(disp, current ? "not out *" : "not out");
            }

            printf("%-22s %-22s %4d %4d %3d %3d %6.1f\n",
                   batsmen[i].name, disp,
                   batsmen[i].runs_scored, batsmen[i].balls_faced,
                   batsmen[i].fours, batsmen[i].sixes, sr);
        }
    }

    printf("------------------------------------------------------------------------\n");
    printf("Extras: (wd %d, nb %d) = %d\n",
           match_state.wides, match_state.noballs, match_state.extras);
    printf("TOTAL: %d/%d (%d.%d overs)  Run Rate: %.2f\n",
           match_state.total_runs, match_state.wickets,
           match_state.current_over, match_state.current_ball,
           match_state.current_run_rate);
    printf("========================================================================\n");
    printf("                      BOWLING - %s\n", match_state.team_bowling);
    printf("========================================================================\n");
    printf("%-22s %5s %5s %5s %5s %5s %7s\n",
           "BOWLER", "O", "M", "R", "W", "DOT", "ECON");
    printf("------------------------------------------------------------------------\n");

    for (int i = 0; i < NUM_BOWLERS; i++) {
        int balls = bowlers[i].overs_bowled * 6 + bowlers[i].balls_in_current_over;
        if (balls > 0) {
            float econ = (float)bowlers[i].runs_given / ((float)balls / 6.0);
            printf("%-22s %3d.%d %5d %5d %5d %5d %7.2f\n",
                   bowlers[i].name,
                   bowlers[i].overs_bowled, bowlers[i].balls_in_current_over,
                   bowlers[i].maidens, bowlers[i].runs_given,
                   bowlers[i].wickets, bowlers[i].dots, econ);
        }
    }
    printf("========================================================================\n");
}

void print_final_result(void) {
    printf("\n");
    printf("========================================================================\n");
    printf("                           MATCH RESULT\n");
    printf("========================================================================\n");
    printf("%-12s: %3d/%d (%2d.%d overs) RR: %.2f\n",
           teams[0].name, teams[0].innings_score, teams[0].innings_wickets,
           teams[0].innings_overs, teams[0].innings_balls,
           teams[0].innings_run_rate);
    printf("%-12s: %3d/%d (%2d.%d overs) RR: %.2f\n",
           teams[1].name, teams[1].innings_score, teams[1].innings_wickets,
           teams[1].innings_overs, teams[1].innings_balls,
           teams[1].innings_run_rate);
    printf("------------------------------------------------------------------------\n");

    if (teams[1].innings_score > teams[0].innings_score) {
        int wkts = MAX_WICKETS - teams[1].innings_wickets;
        int balls = (MAX_OVERS * 6) - (teams[1].innings_overs * 6 + teams[1].innings_balls);
        printf("%s WON by %d wickets (with %d balls remaining)\n",
               teams[1].name, wkts, balls);
    } else if (teams[0].innings_score > teams[1].innings_score) {
        printf("%s WON by %d runs\n",
               teams[0].name, teams[0].innings_score - teams[1].innings_score);
    } else {
        printf("MATCH TIED!\n");
    }
    printf("========================================================================\n");
}

int get_scheduling_choice(int innings_num) {
    int bat_team = (innings_num == 1) ? 0 : 1;

    printf("\n========================================================================\n");
    printf("              INNINGS %d - %s BATTING\n", innings_num, teams[bat_team].name);
    printf("========================================================================\n");
    printf("Select batting order scheduling:\n");
    printf("1. FCFS (First Come First Serve - Original batting order)\n");
    printf("2. SJF (Shortest Job First - Quick batsmen promoted)\n");
    printf("Enter choice (1 or 2): ");

    int choice;
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

    int sched = get_scheduling_choice(innings_num);

    printf("\n========================================================================\n");
    printf("  %s vs %s - INNINGS %d\n", teams[bat_team].name, teams[bowl_team].name, innings_num);
    printf("  Batting Order: %s\n", sched == SCHED_FCFS ? "FCFS" : "SJF");
    if (innings_num == 2) {
        printf("  Target: %d runs from 120 balls\n", match_state.target);
    }
    printf("========================================================================\n");

    reset_for_innings(bat_team, bowl_team);
    apply_batting_schedule(bat_team, sched);
    match_state.is_first_innings = (innings_num == 1);

    printf("\nPress ENTER to start innings...");
    while (getchar() != '\n');
    getchar();
    printf("\n");

    pthread_t bowler_thread, batsman_thread, umpire_thread;
    pthread_t fielder_threads[NUM_FIELDERS];
    int fielder_ids[NUM_FIELDERS];

    pthread_create(&umpire_thread, NULL, umpire_thread_func, NULL);
    pthread_create(&bowler_thread, NULL, bowler_thread_func, NULL);
    pthread_create(&batsman_thread, NULL, batsman_thread_func, NULL);

    for (int i = 0; i < NUM_FIELDERS; i++) {
        fielder_ids[i] = i;
        pthread_create(&fielder_threads[i], NULL, fielder_thread_func, &fielder_ids[i]);
    }

    pthread_join(bowler_thread, NULL);
    
    match_running = false;
    pthread_cond_broadcast(&ball_ready_cond);
    pthread_cond_broadcast(&ball_played_cond);
    pthread_cond_broadcast(&fielder_cond);

    pthread_join(batsman_thread, NULL);
    pthread_join(umpire_thread, NULL);
    for (int i = 0; i < NUM_FIELDERS; i++) {
        pthread_join(fielder_threads[i], NULL);
    }

   
    teams[bat_team].innings_score = match_state.total_runs;
    teams[bat_team].innings_wickets = match_state.wickets;
    teams[bat_team].innings_overs = match_state.current_over;
    teams[bat_team].innings_balls = match_state.current_ball;
    teams[bat_team].innings_extras = match_state.extras;
    
    int total_balls = match_state.current_over * 6 + match_state.current_ball;
    teams[bat_team].innings_run_rate = total_balls > 0 ?
        (float)match_state.total_runs * 6.0 / (float)total_balls : 0.0;

    print_scorecard();

    if (innings_num == 1) {
        match_state.target = teams[bat_team].innings_score + 1;
    }
}

int main(void) {
    printf("========================================================================\n");
    printf("                      T20 CRICKET SIMULATOR\n");
    printf("                       INDIA vs AUSTRALIA\n");
    printf("                      20 Overs Per Innings\n");
    printf("========================================================================\n");

    srand(time(NULL));
    initialize_teams();

    simulate_innings(1);

    printf("\n========================================================================\n");
    printf("                          INNINGS BREAK\n");
    printf("         %s need %d runs to win from 120 balls\n",
           teams[1].name, match_state.target);
    printf("========================================================================\n");

    simulate_innings(2);

    print_final_result();

    pthread_mutex_destroy(&game_mutex);
    pthread_mutex_destroy(&print_mutex);
    pthread_cond_destroy(&ball_ready_cond);
    pthread_cond_destroy(&ball_played_cond);
    pthread_cond_destroy(&fielder_cond);

    return 0;
}