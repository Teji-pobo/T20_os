/*
  T20 Cricket Simulator
  
 Assumptions:
 T20 format so 20 overs per innings and 10 wickets per team 
 In case of a tie 6 overs for powerplay alloted 
 16-20 are death overs so death bowlers are prioritised
 Each bowler has a max of 4 overs and no consecutive overs 
 The pitch is the critical section protected by mutex
 Crease capacity is 2 enforced by semaphore
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

#define TOTAL_OVERS 20
#define BALLS_IN_OVER 6
#define MAX_WKTS 10
#define FIELDER_COUNT 10
#define BOWLER_COUNT 5
#define BATSMAN_COUNT 11
#define QUANTUM 6
#define CREASE_CAP 2
#define DEATH_START 16
#define PP_END 6

#define FCFS_MODE 1
#define SJF_MODE 2

#define OUT_DOT 0
#define OUT_ONE 1
#define OUT_TWO 2
#define OUT_THREE 3
#define OUT_FOUR 4
#define OUT_SIX 5
#define OUT_WICKET 6
#define OUT_WIDE 7
#define OUT_NB 8
int BOWLER_MODE = 0; 
// 0 = Hybrid (current)
// 1 = Pure Round Robin
// 2 = Pure Priority
typedef struct {
    int id;
    char playerName[50];
    int runs;
    int ballsFaced;
    int num4s;
    int num6s;
    bool out;
    int expectedBalls;
    int batsmanPriority;
    bool isTailender;
    char howOut[100];
    int origPosition;
    float sr;
} BatsmanData;

typedef struct {
    int id;
    char bName[50];
    int oversDone;
    int currentOverBalls;
    int runsConceded;
    int wkts;
    int maidenOvers;
    int dotBalls;
    int wideCount;
    int noBallCount;
    bool deathSpecialist;
    bool ppSpecialist;
    int bowlPriority;
    float econ;
} BowlerData;

typedef struct {
    int id;
    char fName[50];
    char pos[30];
    bool isKeeper;
    int catchesTaken;
    int runoutsDone;
    int stumpingsDone;
    bool active;
} FielderData;

typedef struct {
    int strikerID;
    int nonStrikerID;
    int bowlerID;
    char strikerName[50];
    char nonStrikerName[50];
    char bowlerName[50];
    int overNum;
    int ballNum;
    int scoreBefore;
    int wktsBefore;
} BallCtx;

typedef struct {
    int ballNo;
    bool bowled;
    bool played;
    bool inAir;
    int runsOnBall;
    bool wide;
    bool noBall;
    bool isWkt;
    int wktType;
    char desc[300];
    char outcomeStr[50];
    char dismissedName[50];
    int dismissedIdx;
    BallCtx ctx;
} BallData;

typedef struct {
    int totalScore;
    int wktsDown;
    int currOver;
    int currBall;
    int extrasTotal;
    int widesTotal;
    int noBallsTotal;
    int lbTotal;
    int targetRuns;
    bool firstInns;
    int intensity;
    bool matchDone;
    char battingTeam[50];
    char bowlingTeam[50];
    int schedType;
    float runRate;
    float reqRate;
    int ballsLeft;
    int overRuns;
} GameState;

typedef struct {
    int graph[BATSMAN_COUNT][BATSMAN_COUNT];
    bool waiting[BATSMAN_COUNT];
    int waitingFor[BATSMAN_COUNT];
    int holding[BATSMAN_COUNT];
} WaitGraph;

typedef struct {
    int algo;
    int tq;
    int usedQuantum;
    int bQueue[BOWLER_COUNT];
    int front;
    int rear;
    int lastBowler;
} SchedData;

typedef struct {
    int occupant;
    bool locked;
    pthread_mutex_t lck;
} CreaseData;

typedef struct {
    char teamName[50];
    char shortName[10];
    BatsmanData batters[BATSMAN_COUNT];
    BatsmanData origBatters[BATSMAN_COUNT];
    BowlerData bowlers[BOWLER_COUNT];
    FielderData fielders[FIELDER_COUNT];
    int innScore;
    int innWkts;
    int innOvers;
    int innBalls;
    int innExtras;
    float innRR;
} TeamData;

BallData theBall;
GameState game;
SchedData sched;
WaitGraph wfg;
CreaseData crease[2];

TeamData allTeams[2];
int batTeam = 0;
int bowlTeam = 1;

BatsmanData* bats;
BowlerData* bowls;
FielderData* flds;

int onStrike = 0;
int offStrike = 1;
int activeBowler = 0;
int nextBat = 2;

pthread_mutex_t pitchLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t scoreLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t gameLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t ballLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t schedLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t wfgLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t logLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t idxLock = PTHREAD_MUTEX_INITIALIZER;

sem_t creaseSem;
sem_t ballReadySem;
sem_t strokeDoneSem;
sem_t overDoneSem;

pthread_cond_t ballHitCond = PTHREAD_COND_INITIALIZER;
pthread_cond_t ballBowledCond = PTHREAD_COND_INITIALIZER;
pthread_cond_t fielderCond = PTHREAD_COND_INITIALIZER;

volatile bool gameOn = true;
volatile bool ballFlying = false;
volatile bool innDone = false;

FILE* logFile;

int ballCount = 0;

void setupTeams(void);
void setupGame(void);
void setupSched(void);
void setupSync(void);
void setupWFG(void);
void resetInnings(int bt, int bowl);
void setBattingOrder(int t, int mode);
void setFCFS(int t);
void setSJF(int t);

void getContext(BallCtx* c);
void switchBowler(void);
int nextBowlerRR(void);
int nextBowlerPriority(void);

int getBallResult(const BallCtx* c);
void processBall(int result, const BallCtx* c);
void rotateStrike(void);
void bringNewBat(void);
void calcIntensity(void);
void updateRates(void);

void* bowlerFunc(void* arg);
void* batsmanFunc(void* arg);
void* fielderFunc(void* arg);
void* umpireFunc(void* arg);
void* scoreFunc(void* arg);
void* thirdUmpFunc(void* arg);

void addWFGEdge(int from, int to);
void removeWFGEdge(int from, int to);
bool dfsCheck(int node, bool vis[], bool stack[]);
bool hasDeadlock(void);
int fixDeadlock(void);
void handleRunout(const BallCtx* c);

void printBall(const BallCtx* c);
void printOverEnd(void);
void printCard(void);
void printInningSummary(int t);
void printResult(void);
void cleanup(void);

void runInnings(int num);
int getSchedChoice(int num);
 
void setupTeams(void) {
    strcpy(allTeams[0].teamName, "INDIA");
    strcpy(allTeams[0].shortName, "IND");
    

    const char* indBats[] = {
        "Rohit Sharma", "Yashasvi Jaiswal", "Virat Kohli", "Suryakumar Yadav",
        "Hardik Pandya", "Rishabh Pant", "Ravindra Jadeja", "Axar Patel",
        "Kuldeep Yadav", "Jasprit Bumrah", "Mohammed Siraj"
    };
    const int indStay[] = {35, 30, 40, 28, 25, 22, 18, 15, 8, 5, 4};
    
    const char* indBowls[] = {
        "Jasprit Bumrah", "Mohammed Siraj", "Arshdeep Singh",
        "Kuldeep Yadav", "Ravindra Jadeja"
    };
    
    strcpy(allTeams[1].teamName, "AUSTRALIA");
    strcpy(allTeams[1].shortName, "AUS");
    
    const char* ausBats[] = {
        "Travis Head", "David Warner", "Mitchell Marsh", "Steve Smith",
        "Glenn Maxwell", "Marcus Stoinis", "Tim David", "Pat Cummins",
        "Mitchell Starc", "Adam Zampa", "Josh Hazlewood"
    };
    const int ausStay[] = {32, 28, 35, 30, 20, 22, 18, 12, 6, 4, 3};
    
    const char* ausBowls[] = {
        "Pat Cummins", "Mitchell Starc", "Josh Hazlewood",
        "Adam Zampa", "Glenn Maxwell"
    };
    
    const char* fieldPos[] = {
        "Slip", "Gully", "Point", "Cover", "Mid-off",
        "Mid-on", "Midwicket", "Square Leg", "Fine Leg", "Wicketkeeper"
    };
    
    const char** batNames[2] = {indBats, ausBats};
    const char** bowlNames[2] = {indBowls, ausBowls};
    const int* stayArr[2] = {indStay, ausStay};
    
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < BATSMAN_COUNT; i++) {
            allTeams[t].batters[i].id = i;
            allTeams[t].batters[i].origPosition = i;
            strncpy(allTeams[t].batters[i].playerName, batNames[t][i], 49);
            allTeams[t].batters[i].runs = 0;
            allTeams[t].batters[i].ballsFaced = 0;
            allTeams[t].batters[i].num4s = 0;
            allTeams[t].batters[i].num6s = 0;
            allTeams[t].batters[i].out = false;
            allTeams[t].batters[i].expectedBalls = stayArr[t][i];
            allTeams[t].batters[i].batsmanPriority = (i < 4) ? 1 : (i < 7) ? 2 : 3;
            allTeams[t].batters[i].isTailender = (i >= 8);
            allTeams[t].batters[i].sr = 0.0;
            strcpy(allTeams[t].batters[i].howOut, "not out");
            allTeams[t].origBatters[i] = allTeams[t].batters[i];
        }
        
        for (int i = 0; i < BOWLER_COUNT; i++) {
            allTeams[t].bowlers[i].id = i;
            strncpy(allTeams[t].bowlers[i].bName, bowlNames[t][i], 49);
            allTeams[t].bowlers[i].oversDone = 0;
            allTeams[t].bowlers[i].currentOverBalls = 0;
            allTeams[t].bowlers[i].runsConceded = 0;
            allTeams[t].bowlers[i].wkts = 0;
            allTeams[t].bowlers[i].maidenOvers = 0;
            allTeams[t].bowlers[i].dotBalls = 0;
            allTeams[t].bowlers[i].wideCount = 0;
            allTeams[t].bowlers[i].noBallCount = 0;
            allTeams[t].bowlers[i].deathSpecialist = (i == 0 || i == 1);
            allTeams[t].bowlers[i].ppSpecialist = (i == 1 || i == 2);
            allTeams[t].bowlers[i].bowlPriority = allTeams[t].bowlers[i].deathSpecialist ? 1 : 2;
            allTeams[t].bowlers[i].econ = 0.0;
        }
        
        for (int i = 0; i < FIELDER_COUNT; i++) {
            allTeams[t].fielders[i].id = i;
            snprintf(allTeams[t].fielders[i].fName, 50, "Fielder %d", i + 1);
            strncpy(allTeams[t].fielders[i].pos, fieldPos[i], 29);
            allTeams[t].fielders[i].isKeeper = (i == 9);
            allTeams[t].fielders[i].catchesTaken = 0;
            allTeams[t].fielders[i].runoutsDone = 0;
            allTeams[t].fielders[i].stumpingsDone = 0;
            allTeams[t].fielders[i].active = false;
        }
        
        allTeams[t].innScore = 0;
        allTeams[t].innWkts = 0;
        allTeams[t].innOvers = 0;
        allTeams[t].innBalls = 0;
        allTeams[t].innExtras = 0;
        allTeams[t].innRR = 0.0;
    }
}

void setupGame(void) {
    game.totalScore = 0;
    game.wktsDown = 0;
    game.currOver = 0;
    game.currBall = 0;
    game.extrasTotal = 0;
    game.widesTotal = 0;
    game.noBallsTotal = 0;
    game.lbTotal = 0;
    game.targetRuns = 0;
    game.firstInns = true;
    game.intensity = 0;
    game.matchDone = false;
    game.schedType = FCFS_MODE;
    game.runRate = 0.0;
    game.reqRate = 0.0;
    game.ballsLeft = TOTAL_OVERS * BALLS_IN_OVER;
    game.overRuns = 0;
    
    memset(&theBall, 0, sizeof(BallData));
    theBall.dismissedIdx = -1;
    ballCount = 0;
}

void setupSched(void) {
    sched.algo = 0;
    sched.tq = QUANTUM;
    sched.usedQuantum = 0;
    sched.lastBowler = -1;
    for (int i = 0; i < BOWLER_COUNT; i++) sched.bQueue[i] = i;
    sched.front = 0;
    sched.rear = BOWLER_COUNT - 1;
}

void setupSync(void) {
    sem_init(&creaseSem, 0, CREASE_CAP);
    sem_init(&ballReadySem, 0, 0);
    sem_init(&strokeDoneSem, 0, 0);
    sem_init(&overDoneSem, 0, 0);
    for (int i = 0; i < 2; i++) {
        crease[i].occupant = -1;
        crease[i].locked = false;
        pthread_mutex_init(&crease[i].lck, NULL);
    }
}

void setupWFG(void) {
    for (int i = 0; i < BATSMAN_COUNT; i++) {
        wfg.waiting[i] = false;
        wfg.waitingFor[i] = -1;
        wfg.holding[i] = -1;
        for (int j = 0; j < BATSMAN_COUNT; j++) wfg.graph[i][j] = 0;
    }
}
void resetInnings(int bt, int bowl) {
    batTeam = bt;
    bowlTeam = bowl;
    
    bats = allTeams[bt].batters;
    bowls = allTeams[bowl].bowlers;
    flds = allTeams[bowl].fielders;
    
    for (int i = 0; i < BATSMAN_COUNT; i++) {
        bats[i].runs = 0;
        bats[i].ballsFaced = 0;
        bats[i].num4s = 0;
        bats[i].num6s = 0;
        bats[i].out = false;
        bats[i].sr = 0.0;
        strcpy(bats[i].howOut, "not out");
    }
    
    for (int i = 0; i < BOWLER_COUNT; i++) {
        bowls[i].oversDone = 0;
        bowls[i].currentOverBalls = 0;
        bowls[i].runsConceded = 0;
        bowls[i].wkts = 0;
        bowls[i].maidenOvers = 0;
        bowls[i].dotBalls = 0;
        bowls[i].wideCount = 0;
        bowls[i].noBallCount = 0;
        bowls[i].econ = 0.0;
    }
    
    onStrike = 0;
    offStrike = 1;
    activeBowler = 0;
    nextBat = 2;
    
    game.totalScore = 0;
    game.wktsDown = 0;
    game.currOver = 0;
    game.currBall = 0;
    game.extrasTotal = 0;
    game.widesTotal = 0;
    game.noBallsTotal = 0;
    game.lbTotal = 0;
    game.intensity = 0;
    game.runRate = 0.0;
    game.ballsLeft = TOTAL_OVERS * BALLS_IN_OVER;
    game.overRuns = 0;
    
    strcpy(game.battingTeam, allTeams[bt].teamName);
    strcpy(game.bowlingTeam, allTeams[bowl].teamName);
    
    sched.usedQuantum = 0;
    sched.front = 0;
    sched.lastBowler = -1;
    
    crease[0].occupant = onStrike;
    crease[1].occupant = offStrike;
    
    setupWFG(); 
    wfg.holding[onStrike] = 0;
    wfg.holding[offStrike] = 1;

    int val;
    sem_getvalue(&ballReadySem, &val);
    while (val > 0) { sem_wait(&ballReadySem); sem_getvalue(&ballReadySem, &val); }
    sem_getvalue(&strokeDoneSem, &val);
    while (val > 0) { sem_wait(&strokeDoneSem); sem_getvalue(&strokeDoneSem, &val); }
    
    memset(&theBall, 0, sizeof(BallData));
    theBall.dismissedIdx = -1;
    
    gameOn = true;
    innDone = false;
    ballFlying = false;
    ballCount = 0;
}

void setFCFS(int t) {
    printf("Using FCFS - Original batting order\n");
    for (int i = 0; i < BATSMAN_COUNT; i++) {
        allTeams[t].batters[i] = allTeams[t].origBatters[i];
        allTeams[t].batters[i].runs = 0;
        allTeams[t].batters[i].ballsFaced = 0;
        allTeams[t].batters[i].num4s = 0;
        allTeams[t].batters[i].num6s = 0;
        allTeams[t].batters[i].out = false;
        allTeams[t].batters[i].sr = 0.0;
        strcpy(allTeams[t].batters[i].howOut, "not out");
    }
}

void setSJF(int t) {
    printf("Using SJF - Sorted by expected stay\n");
    for (int i = 0; i < BATSMAN_COUNT; i++) {
        allTeams[t].batters[i] = allTeams[t].origBatters[i];
        allTeams[t].batters[i].runs = 0;
        allTeams[t].batters[i].ballsFaced = 0;
        allTeams[t].batters[i].num4s = 0;
        allTeams[t].batters[i].num6s = 0;
        allTeams[t].batters[i].out = false;
        allTeams[t].batters[i].sr = 0.0;
        strcpy(allTeams[t].batters[i].howOut, "not out");
    }
    for (int i = 2; i < BATSMAN_COUNT - 1; i++) {
        for (int j = i + 1; j < BATSMAN_COUNT; j++) {
            if (allTeams[t].batters[j].expectedBalls < allTeams[t].batters[i].expectedBalls) {
                BatsmanData tmp = allTeams[t].batters[i];
                allTeams[t].batters[i] = allTeams[t].batters[j];
                allTeams[t].batters[j] = tmp;
            }
        }
    }
}

void setBattingOrder(int t, int mode) {
    game.schedType = mode;
    if (mode == FCFS_MODE) setFCFS(t);
    else setSJF(t);
}

void getContext(BallCtx* c) {
    pthread_mutex_lock(&idxLock);
    c->strikerID = onStrike;
    c->nonStrikerID = offStrike;
    c->bowlerID = activeBowler;
    strncpy(c->strikerName, bats[onStrike].playerName, 49);
    strncpy(c->nonStrikerName, bats[offStrike].playerName, 49);
    strncpy(c->bowlerName, bowls[activeBowler].bName, 49);
    c->overNum = game.currOver;
    c->ballNum = game.currBall;
    c->scoreBefore = game.totalScore;
    c->wktsBefore = game.wktsDown;
    pthread_mutex_unlock(&idxLock);
}

int nextBowlerRR(void) {
    int next;
    int tries = 0;
    int prev = activeBowler;
    do {
        sched.front = (sched.front + 1) % BOWLER_COUNT;
        next = sched.bQueue[sched.front];
        tries++;
        if (bowls[next].oversDone < 4 && next != prev) break;
    } while (tries < BOWLER_COUNT * 2);
    return next;
}

int nextBowlerPriority(void) {
    int best = -1;
    int highPri = 999;
    for (int i = 0; i < BOWLER_COUNT; i++) {
        if (bowls[i].oversDone < 4 && i != activeBowler) {
            int pri = bowls[i].bowlPriority;
            if (game.currOver >= DEATH_START && bowls[i].deathSpecialist) pri = 0;
            else if (game.currOver < PP_END && bowls[i].ppSpecialist) pri = 0;
            if (bowls[i].oversDone > 0) {
                float e = (float)bowls[i].runsConceded / ((float)(bowls[i].oversDone * 6 + bowls[i].currentOverBalls) / 6.0);
                if (e < 6.0) pri--;
            }
            if (pri < highPri) {
                highPri = pri;
                best = i;
            }
        }
    }
    return (best == -1) ? nextBowlerRR() : best;
}

void printOverEnd(void) {
    pthread_mutex_lock(&logLock);
    int b = activeBowler;
    bool maiden = (game.overRuns == 0 && bowls[b].currentOverBalls == 6);
    if (maiden) bowls[b].maidenOvers++;
    printf("End of Over %d: %s %d-%d-%d-%d %s | %s %d/%d | RR: %.2f\n",
           game.currOver + 1, bowls[b].bName, bowls[b].oversDone + 1,
           bowls[b].maidenOvers, bowls[b].runsConceded, bowls[b].wkts,
           maiden ? "(Maiden)" : "", game.battingTeam, game.totalScore,
           game.wktsDown, game.runRate);
    pthread_mutex_unlock(&logLock);
}
// gant chart
void switchBowler(void) {
    pthread_mutex_lock(&schedLock);
    pthread_mutex_lock(&idxLock);
    int old = activeBowler;
    bool maiden = (game.overRuns == 0);
    if (maiden && bowls[old].currentOverBalls >= 6) bowls[old].maidenOvers++;
    bowls[old].oversDone++;
    int totalB = bowls[old].oversDone * 6;
    if (totalB > 0) bowls[old].econ = (float)bowls[old].runsConceded / ((float)totalB / 6.0);
    pthread_mutex_unlock(&idxLock);
    pthread_mutex_unlock(&schedLock);
    
    printOverEnd();
    
    pthread_mutex_lock(&schedLock);
    pthread_mutex_lock(&idxLock);
    // changes for the gant chart 
    // start

if (BOWLER_MODE == 1) {
                        // Pure Round Robin
    sched.algo = 0;
    activeBowler = nextBowlerRR();
}
else if (BOWLER_MODE == 2) {
// Pure Priority
    sched.algo = 2;
    activeBowler = nextBowlerPriority();
}
else {
    // Hybrid Logic
    if (game.currOver >= DEATH_START - 1 || game.intensity > 70) 
        sched.algo = 2;
    else if (game.currOver < PP_END) 
        sched.algo = 2;
    else 
        sched.algo = 0;

    activeBowler = (sched.algo == 2) ? nextBowlerPriority() : nextBowlerRR();
}

    //end
    sched.lastBowler = old;
    bowls[activeBowler].currentOverBalls = 0;
    
    int tmp = onStrike;
    onStrike = offStrike;
    offStrike = tmp;
    crease[0].occupant = onStrike;
    crease[1].occupant = offStrike;
    
    game.currOver++;
    game.currBall = 0;
    game.overRuns = 0;
    sched.usedQuantum = 0;
    pthread_mutex_unlock(&idxLock);
    pthread_mutex_unlock(&schedLock);
}

void addWFGEdge(int from, int to) {
    pthread_mutex_lock(&wfgLock);
    wfg.graph[from][to] = 1;
    wfg.waiting[from] = true;
    pthread_mutex_unlock(&wfgLock);
}

void removeWFGEdge(int from, int to) {
    pthread_mutex_lock(&wfgLock);
    wfg.graph[from][to] = 0;
    wfg.waiting[from] = false;
    pthread_mutex_unlock(&wfgLock);
}

bool dfsCheck(int node, bool vis[], bool stack[]) {
    vis[node] = true;
    stack[node] = true;
    for (int i = 0; i < BATSMAN_COUNT; i++) {
        if (wfg.graph[node][i]) {
            if (!vis[i] && dfsCheck(i, vis, stack)) return true;
            else if (stack[i]) return true;
        }
    }
    stack[node] = false;
    return false;
}

bool hasDeadlock(void) {
    pthread_mutex_lock(&wfgLock);
    bool vis[BATSMAN_COUNT] = {false};
    bool stack[BATSMAN_COUNT] = {false};
    for (int i = 0; i < BATSMAN_COUNT; i++) {
        if (wfg.waiting[i] && !vis[i]) {
            if (dfsCheck(i, vis, stack)) {
                pthread_mutex_unlock(&wfgLock);
                return true;
            }
        }
    }
    pthread_mutex_unlock(&wfgLock);
    return false;
}

int fixDeadlock(void) {
    printf("Deadlock detected - Third Umpire reviewing\n");
    int victim = -1;
    pthread_mutex_lock(&wfgLock);
    for (int i = 0; i < BATSMAN_COUNT; i++) {
        if (wfg.waiting[i]) {
            if (victim == -1 || bats[i].runs < bats[victim].runs) victim = i;
        }
    }
    for (int i = 0; i < BATSMAN_COUNT; i++) {
        wfg.waiting[i] = false;
        wfg.waitingFor[i] = -1;
        for (int j = 0; j < BATSMAN_COUNT; j++) wfg.graph[i][j] = 0;
    }
    pthread_mutex_unlock(&wfgLock);
    
    if (victim != -1) {
        bats[victim].out = true;
        snprintf(bats[victim].howOut, 100, "run out (Deadlock)");
        printf("OUT! %s run out (Deadlock resolved)\n", bats[victim].playerName);
    }
    return victim;
}

void handleRunout(const BallCtx* c) {
    if (theBall.isWkt) return;                     // ✅ already out → ignore
    if (theBall.runsOnBall < 1 || theBall.runsOnBall > 3) return;   // ✅ only 1–3 runs   

    if (theBall.isWkt) return;   // ✅ prevent double dismissal

    int runs = theBall.runsOnBall;
    if (runs <= 0 || runs >= 4) return;
    
    pthread_mutex_lock(&idxLock);
    int str = onStrike;
    int nstr = offStrike;
    pthread_mutex_unlock(&idxLock);
    
    for (int r = 0; r < runs; r++) {
        addWFGEdge(str, nstr);
        addWFGEdge(nstr, str);

        if (hasDeadlock()) {
            int prob = 12 + (game.intensity / 10);
            if (rand() % 100 < prob) {
                int out = fixDeadlock();
                if (out != -1) {
                    pthread_mutex_lock(&scoreLock);
                    game.wktsDown++;
                    pthread_mutex_unlock(&scoreLock);

                    theBall.isWkt = true;
                    theBall.wktType = 4;
                    theBall.dismissedIdx = out;
                    strncpy(theBall.dismissedName, bats[out].playerName, 49);

                    // bringNewBat();
                    break;
                }
            }
        }

        removeWFGEdge(str, nstr);
        removeWFGEdge(nstr, str);
    }
    
    if (runs % 2 == 1 && !theBall.isWkt) {
        pthread_mutex_lock(&idxLock);
        int tmp = onStrike;
        onStrike = offStrike;
        offStrike = tmp;
        pthread_mutex_unlock(&idxLock);
    }
}

int getBallResult(const BallCtx* c) {
    int rv = rand() % 1000;
    int striker = c->strikerID;
    int bowler = c->bowlerID;
    
    int batSkill = 700 - (bats[striker].isTailender ? 250 : 0);
    int bowlSkill = 600 + (bowls[bowler].deathSpecialist ? 80 : 0);
    if (game.intensity > 70) batSkill += 50;
    if (game.currOver >= DEATH_START) { batSkill += 30; bowlSkill += 20; }
    if (game.currOver < PP_END) batSkill += 40;
    if (!game.firstInns && game.targetRuns > 0) {
        int need = game.targetRuns - game.totalScore;
        if (game.reqRate > 12.0) batSkill += 60;
        if (need <= 20 && game.ballsLeft <= 12) batSkill += 80;
    }
    
    if (rv < 25) return OUT_WIDE;
    if (rv < 40) return OUT_NB;
    
    int wktProb = 50 + (bowlSkill - batSkill) / 8;
    if (wktProb < 30) wktProb = 30;
    if (wktProb > 120) wktProb = 120;
    if (rv < 40 + wktProb) return OUT_WICKET;
    
    int boundary = (game.currOver >= DEATH_START) ? 30 : 0;
    int pp = (game.currOver < PP_END) ? 25 : 0;
    if (rv < 340) return OUT_DOT;
    if (rv < 590) return OUT_ONE;
    if (rv < 710) return OUT_TWO;
    if (rv < 740) return OUT_THREE;
    if (rv < 890 + boundary + pp) return OUT_FOUR;
    return OUT_SIX;
}

void processBall(int result, const BallCtx* c) {
    pthread_mutex_lock(&scoreLock);
    int striker = c->strikerID;
    int bowler = c->bowlerID;
    int runs = 0;
    bool wkt = false;
    bool extra = false;
    bool legal = true;
    theBall.dismissedIdx = -1;
    strcpy(theBall.dismissedName, "");
    strcpy(theBall.outcomeStr, "");
    
    switch (result) {
        case OUT_DOT:
            runs = 0;
            strcpy(theBall.outcomeStr, ".");
            bowls[bowler].dotBalls++;
            break;
        case OUT_ONE:
            runs = 1;
            strcpy(theBall.outcomeStr, "1");
            break;
        case OUT_TWO:
            runs = 2;
            strcpy(theBall.outcomeStr, "2");
            break;
        case OUT_THREE:
            runs = 3;
            strcpy(theBall.outcomeStr, "3");
            break;
        case OUT_FOUR:
            runs = 4;
            strcpy(theBall.outcomeStr, "4");
            bats[striker].num4s++;
            break;
        case OUT_SIX:
            runs = 6;
            strcpy(theBall.outcomeStr, "6");
            bats[striker].num6s++;
            ballFlying = true;
            break;
        case OUT_WICKET: {
            runs = 0;
            wkt = true;
            int wtype = rand() % 5;
            theBall.wktType = wtype;
            theBall.dismissedIdx = striker;
            strncpy(theBall.dismissedName, c->strikerName, 49);
            const char* wtypes[] = {"Bowled", "Caught", "LBW", "Stumped", "C&B"};
            strcpy(theBall.outcomeStr, wtypes[wtype]);
            if (wtype == 1 || wtype == 4) ballFlying = true;
            const char* fmt[] = {"b %s", "c sub b %s", "lbw b %s", "st b %s", "c & b %s"};
            snprintf(bats[striker].howOut, 100, fmt[wtype], c->bowlerName);
            bats[striker].out = true;
            break;
        }
        case OUT_WIDE:
            runs = 1;
            extra = true;
            legal = false;
            strcpy(theBall.outcomeStr, "Wd");
            bowls[bowler].wideCount++;
            game.widesTotal++;
            theBall.wide = true;
            break;
        case OUT_NB:
            runs = 1;
            extra = true;
            legal = false;
            strcpy(theBall.outcomeStr, "Nb");
            bowls[bowler].noBallCount++;
            game.noBallsTotal++;
            theBall.noBall = true;
            break;
    }
    
    game.totalScore += runs;
    game.overRuns += runs;
    
    if (!extra) {
        bats[striker].runs += runs;
        if (legal) {
            bats[striker].ballsFaced++;
            if (bats[striker].ballsFaced > 0)
                bats[striker].sr = (float)bats[striker].runs * 100.0 / (float)bats[striker].ballsFaced;
        }
    } else {
        game.extrasTotal += runs;
    }
    
    bowls[bowler].runsConceded += runs;
    theBall.runsOnBall = runs;
    theBall.isWkt = wkt;
    
    snprintf(theBall.desc, sizeof(theBall.desc), "%d.%d %s to %s, %s",
             c->overNum, c->ballNum + 1, c->bowlerName, c->strikerName, theBall.outcomeStr);
    
    if (wkt) {
        game.wktsDown++;
        bowls[bowler].wkts++;
    }
    
    int totalB = bowls[bowler].oversDone * 6 + bowls[bowler].currentOverBalls + 1;
    if (totalB > 0) bowls[bowler].econ = (float)bowls[bowler].runsConceded / ((float)totalB / 6.0);
    memcpy(&theBall.ctx, c, sizeof(BallCtx));
    pthread_mutex_unlock(&scoreLock);
}

void rotateStrike(void) {
    pthread_mutex_lock(&idxLock);
    int tmp = onStrike;
    onStrike = offStrike;
    offStrike = tmp;
    crease[0].occupant = onStrike;
    crease[1].occupant = offStrike;
    pthread_mutex_unlock(&idxLock);
}

void bringNewBat(void) {
    pthread_mutex_lock(&idxLock);
    int newB = nextBat;
    while (newB < BATSMAN_COUNT && bats[newB].out) newB++;
    if (newB < BATSMAN_COUNT && !bats[newB].out) {
        if (bats[onStrike].out) onStrike = newB;
        else if (bats[offStrike].out) offStrike = newB;
        nextBat = newB + 1;
        crease[0].occupant = onStrike;
        crease[1].occupant = offStrike;
        printf("New batsman: %s\n", bats[newB].playerName);
    }
    pthread_mutex_unlock(&idxLock);
}

void calcIntensity(void) {
    int i = 0;
    if (game.currOver >= DEATH_START) i += 30;
    if (game.currOver >= TOTAL_OVERS - 2) i += 25;
    if (game.currOver == TOTAL_OVERS - 1) i += 20;
    if (game.wktsDown >= 5) i += 15;
    if (game.wktsDown >= 7) i += 15;
    if (game.wktsDown >= 9) i += 20;
    if (!game.firstInns && game.targetRuns > 0) {
        int rem = game.targetRuns - game.totalScore;
        int bLeft = (TOTAL_OVERS - game.currOver) * 6 - game.currBall;
        if (bLeft > 0) {
            float rrr = (float)rem * 6.0 / (float)bLeft;
            if (rrr > 10.0) i += 25;
            if (rrr > 12.0) i += 15;
        }
        if (rem <= 30 && bLeft <= 18) i += 20;
        if (rem <= 15 && bLeft <= 12) i += 25;
    }
    game.intensity = (i > 100) ? 100 : i;
}

void updateRates(void) {
    int tb = game.currOver * 6 + game.currBall;
    if (tb > 0) game.runRate = (float)game.totalScore * 6.0 / (float)tb;
    game.ballsLeft = (TOTAL_OVERS * 6) - tb;
    if (!game.firstInns && game.targetRuns > 0 && game.ballsLeft > 0) {
        int need = game.targetRuns - game.totalScore;
        game.reqRate = (float)need * 6.0 / (float)game.ballsLeft;
    }
}

void* bowlerFunc(void* arg) {
    int id = *(int*)arg;
    while (gameOn) {
        pthread_mutex_lock(&schedLock);
        int active = activeBowler;
        pthread_mutex_unlock(&schedLock);
        if (active != id) { usleep(30000); continue; }
        if (bowls[id].oversDone >= 4) { switchBowler(); continue; }
        
        BallCtx ctx;
        getContext(&ctx);
        
        pthread_mutex_lock(&pitchLock);
        theBall.ballNo++;
        theBall.bowled = true;
        theBall.played = false;
        theBall.inAir = false;
        theBall.wide = false;
        theBall.noBall = false;
        theBall.isWkt = false;
        theBall.dismissedIdx = -1;
        memcpy(&theBall.ctx, &ctx, sizeof(BallCtx));
        pthread_mutex_unlock(&pitchLock);
        
        sem_post(&ballReadySem);
        sem_wait(&strokeDoneSem);
        
        pthread_mutex_lock(&schedLock);
        if (!theBall.wide && !theBall.noBall) {
            bowls[id].currentOverBalls++;
            game.currBall++;
            sched.usedQuantum++;
            ballCount++;
        }
        pthread_mutex_unlock(&schedLock);
        
        printBall(&theBall.ctx);
        updateRates();
        
        if (sched.usedQuantum >= QUANTUM) {
        switchBowler();
        continue;  
        }
        
        if (game.wktsDown >= MAX_WKTS || game.currOver >= TOTAL_OVERS) {
            innDone = true;
            gameOn = false;
        }
        if (!game.firstInns && game.totalScore >= game.targetRuns) {
            innDone = true;
            gameOn = false;
        }
        usleep(120000);
    }
    return NULL;
}

void* batsmanFunc(void* arg) {
    int id = *(int*)arg;
    while (gameOn) {
        pthread_mutex_lock(&idxLock);
        bool isStr = (id == onStrike);
        bool isNstr = (id == offStrike);
        pthread_mutex_unlock(&idxLock);
        if (!isStr && !isNstr) { usleep(30000); continue; }
        if (!isStr) { usleep(15000); continue; }
        
        sem_wait(&ballReadySem);
        if (!gameOn) break;
        
       BallCtx ctx;
getContext(&ctx); 

pthread_mutex_lock(&pitchLock);

// ✅ STRICT CONTROL: only striker + only once
if (theBall.bowled && !theBall.played && id == onStrike) {
    int result = getBallResult(&ctx);
    processBall(result, &ctx);
    theBall.played = true;

    // ✅ handle everything here (single thread)
    if (theBall.runsOnBall >= 1 && theBall.runsOnBall <= 3 && !theBall.isWkt) {
        handleRunout(&ctx);
    }
    else if (theBall.runsOnBall > 0 && !theBall.isWkt) {
        if (theBall.runsOnBall % 2 == 1) rotateStrike();
    }

    if (theBall.isWkt && game.wktsDown < MAX_WKTS) {
        bringNewBat();
        theBall.isWkt = false;
    }

    if (ballFlying) {
        pthread_mutex_lock(&ballLock);
        pthread_cond_broadcast(&ballHitCond);
        pthread_mutex_unlock(&ballLock);
    }
}

pthread_mutex_unlock(&pitchLock);
     
        calcIntensity();
        sem_post(&strokeDoneSem);
    }
    return NULL;
}

void* fielderFunc(void* arg) {
    int id = *(int*)arg;
    while (gameOn) {
        pthread_mutex_lock(&ballLock);
        while (!ballFlying && gameOn) pthread_cond_wait(&ballHitCond, &ballLock);
        if (!gameOn) { pthread_mutex_unlock(&ballLock); break; }
        flds[id].active = true;
        pthread_mutex_unlock(&ballLock);
        
        usleep(30000);
        
        if (theBall.isWkt && (theBall.wktType == 1 || theBall.wktType == 4)) {
            if (rand() % 100 < 75) flds[id].catchesTaken++;
        }
        
        pthread_mutex_lock(&ballLock);
        ballFlying = false;
        flds[id].active = false;
        pthread_mutex_unlock(&ballLock);
    }
    return NULL;
}

void* umpireFunc(void* arg) {
    (void)arg;
    while (gameOn) {
        // if (hasDeadlock()) fixDeadlock();
        pthread_mutex_lock(&gameLock);
        if (game.wktsDown >= MAX_WKTS) {
            printf("ALL OUT!\n");
            gameOn = false;
        }
        if (game.currOver >= TOTAL_OVERS) {
            printf("20 OVERS COMPLETE!\n");
            gameOn = false;
        }
        if (!game.firstInns && game.totalScore >= game.targetRuns) {
            printf("TARGET ACHIEVED!\n");
            gameOn = false;
        }
        pthread_mutex_unlock(&gameLock);
        usleep(80000);
    }
    return NULL;
}

void* scoreFunc(void* arg) {
    (void)arg;
    while (gameOn) {
        pthread_mutex_lock(&scoreLock);
        pthread_mutex_lock(&idxLock);
        int s = onStrike;
        int ns = offStrike;
        printf("\r%s: %d/%d (%d.%d) %s %d(%d)* %s %d(%d) RR: %.2f",
               game.battingTeam, game.totalScore, game.wktsDown,
               game.currOver, game.currBall,
               bats[s].playerName, bats[s].runs, bats[s].ballsFaced,
               bats[ns].playerName, bats[ns].runs, bats[ns].ballsFaced,
               game.runRate);
        if (!game.firstInns && game.targetRuns > 0) {
            int need = game.targetRuns - game.totalScore;
            printf(" Need %d off %d (RRR: %.2f)", need, game.ballsLeft, game.reqRate);
        }
        printf("   ");
        fflush(stdout);
        pthread_mutex_unlock(&idxLock);
        pthread_mutex_unlock(&scoreLock);
        sleep(1);
    }
    return NULL;
}

void* thirdUmpFunc(void* arg) {
    (void)arg;
    while (gameOn) {
        pthread_mutex_lock(&schedLock);
        if (game.currOver >= DEATH_START && sched.algo != 2) sched.algo = 2;
        pthread_mutex_unlock(&schedLock);
        
        pthread_mutex_lock(&wfgLock);
        bool anyWait = false;
        for (int i = 0; i < BATSMAN_COUNT; i++) if (wfg.waiting[i]) { anyWait = true; break; }
        if (anyWait) {
            usleep(150000);
            // if (hasDeadlock()) fixDeadlock();
        }
        pthread_mutex_unlock(&wfgLock);
        usleep(200000);
    }
    return NULL;
}

void printBall(const BallCtx* c) {
    pthread_mutex_lock(&logLock);
    printf("\n%s", theBall.desc);
    if (theBall.isWkt) {
        const char* dname = strlen(theBall.dismissedName) > 0 ? theBall.dismissedName : c->strikerName;
        int didx = theBall.dismissedIdx >= 0 ? theBall.dismissedIdx : c->strikerID;
        printf("\nOUT! %s %d(%d) - %s", dname, bats[didx].runs, bats[didx].ballsFaced, bats[didx].howOut);
    }
    if (theBall.runsOnBall == 4) printf(" FOUR!");
    else if (theBall.runsOnBall == 6) printf(" SIX!");
    printf("\n");
    fflush(stdout);
    pthread_mutex_unlock(&logLock);
}

void printCard(void) {
    printf("\n--- SCORECARD - %s ---\n", game.battingTeam);
    printf("%-25s %-25s %4s %4s %3s %3s %6s\n", "BATSMAN", "DISMISSAL", "R", "B", "4", "6", "SR");
    for (int i = 0; i < BATSMAN_COUNT; i++) {
        if (bats[i].ballsFaced > 0 || bats[i].out) {
            bool curr = (i == onStrike || i == offStrike) && !bats[i].out;
            float sr = bats[i].ballsFaced > 0 ? (float)bats[i].runs * 100.0 / bats[i].ballsFaced : 0.0;
            printf("%-25s %-25s %4d %4d %3d %3d %6.1f\n",
                   bats[i].playerName,
                   bats[i].out ? bats[i].howOut : (curr ? "not out *" : "not out"),
                   bats[i].runs, bats[i].ballsFaced, bats[i].num4s, bats[i].num6s, sr);
        }
    }
    printf("Extras: (w %d, nb %d) %d\n", game.widesTotal, game.noBallsTotal, game.extrasTotal);
    printf("TOTAL: %d/%d (%d.%d overs) RR: %.2f\n\n",
           game.totalScore, game.wktsDown, game.currOver, game.currBall, game.runRate);
    
    printf("--- BOWLING - %s ---\n", game.bowlingTeam);
    printf("%-25s %5s %5s %5s %5s %6s\n", "BOWLER", "O", "M", "R", "W", "ECON");
    for (int i = 0; i < BOWLER_COUNT; i++) {
        int tb = bowls[i].oversDone * 6 + bowls[i].currentOverBalls;
        if (tb > 0) {
            float eco = (float)bowls[i].runsConceded / ((float)tb / 6.0);
            printf("%-25s %3d.%d %5d %5d %5d %6.2f\n",
                   bowls[i].bName, bowls[i].oversDone, bowls[i].currentOverBalls,
                   bowls[i].maidenOvers, bowls[i].runsConceded, bowls[i].wkts, eco);
        }
    }
    printf("\n");
}

void printInningSummary(int t) {
    printf("\nINNINGS %d COMPLETE\n", t + 1);
    printf("%s: %d/%d in %d.%d overs (RR: %.2f)\n",
           allTeams[t].teamName, allTeams[t].innScore, allTeams[t].innWkts,
           allTeams[t].innOvers, allTeams[t].innBalls, allTeams[t].innRR);
    printf("Batting Schedule: %s\n\n", game.schedType == FCFS_MODE ? "FCFS" : "SJF");
}

void printResult(void) {
    printf("\n    MATCH RESULT    \n");
    printf("%s: %d/%d (%d.%d overs)\n",
           allTeams[0].teamName, allTeams[0].innScore, allTeams[0].innWkts,
           allTeams[0].innOvers, allTeams[0].innBalls);
    printf("%s: %d/%d (%d.%d overs)\n",
           allTeams[1].teamName, allTeams[1].innScore, allTeams[1].innWkts,
           allTeams[1].innOvers, allTeams[1].innBalls);
    
    if (allTeams[1].innScore > allTeams[0].innScore) {
        int wktsRem = MAX_WKTS - allTeams[1].innWkts;
        int ballsRem = (TOTAL_OVERS * 6) - (allTeams[1].innOvers * 6 + allTeams[1].innBalls);
        printf("%s WINS by %d wickets (with %d balls remaining)\n",
               allTeams[1].teamName, wktsRem, ballsRem);
    } else if (allTeams[0].innScore > allTeams[1].innScore) {
        int margin = allTeams[0].innScore - allTeams[1].innScore;
        printf("%s WINS by %d runs\n", allTeams[0].teamName, margin);
    } else {
        printf("MATCH TIED\n");
    }
}

void cleanup(void) {
    pthread_mutex_destroy(&pitchLock);
    pthread_mutex_destroy(&scoreLock);
    pthread_mutex_destroy(&gameLock);
    pthread_mutex_destroy(&ballLock);
    pthread_mutex_destroy(&schedLock);
    pthread_mutex_destroy(&wfgLock);
    pthread_mutex_destroy(&logLock);
    pthread_mutex_destroy(&idxLock);
    sem_destroy(&creaseSem);
    sem_destroy(&ballReadySem);
    sem_destroy(&strokeDoneSem);
    sem_destroy(&overDoneSem);
    pthread_cond_destroy(&ballHitCond);
    pthread_cond_destroy(&ballBowledCond);
    pthread_cond_destroy(&fielderCond);
    for (int i = 0; i < 2; i++) pthread_mutex_destroy(&crease[i].lck);
    if (logFile) fclose(logFile);
}

int getSchedChoice(int num) {
    int choice;
    int bt = (num == 1) ? 0 : 1;
    printf("\nSelect batting schedule for Innings %d (%s):\n", num, allTeams[bt].teamName);
    printf("1. FCFS (First Come First Serve)\n");
    printf("2. SJF (Shortest Job First)\n");
    printf("Enter choice (1 or 2): ");
    if (scanf("%d", &choice) != 1 || (choice != 1 && choice != 2)) {
        printf("Invalid. Using FCFS.\n");
        choice = 1;
        while (getchar() != '\n');
    }
    return choice;
}

void runInnings(int num) {
    int bt = (num == 1) ? 0 : 1;
    int bowl = (num == 1) ? 1 : 0;
    int schedChoice = getSchedChoice(num);
    
    printf("\nInnings %d: %s batting, %s bowling\n", num, allTeams[bt].teamName, allTeams[bowl].teamName);
    if (num == 2) printf("Target: %d\n", game.targetRuns);
    
    resetInnings(bt, bowl);
    setBattingOrder(bt, schedChoice);
    game.firstInns = (num == 1);
    
    printf("Press ENTER to start  ");
    while (getchar() != '\n');
    getchar();
    
    pthread_t bowlThreads[BOWLER_COUNT];
    pthread_t batThreads[BATSMAN_COUNT];
    pthread_t fieldThreads[FIELDER_COUNT];
    pthread_t umpThread, scoreThread, thirdThread;
    
    int bowlIds[BOWLER_COUNT];
    int batIds[BATSMAN_COUNT];
    int fieldIds[FIELDER_COUNT];
    
    pthread_create(&umpThread, NULL, umpireFunc, NULL);
    pthread_create(&thirdThread, NULL, thirdUmpFunc, NULL);
    pthread_create(&scoreThread, NULL, scoreFunc, NULL);
    
    for (int i = 0; i < BOWLER_COUNT; i++) {
        bowlIds[i] = i;
        pthread_create(&bowlThreads[i], NULL, bowlerFunc, &bowlIds[i]);
    }
    for (int i = 0; i < BATSMAN_COUNT; i++) {
        batIds[i] = i;
        pthread_create(&batThreads[i], NULL, batsmanFunc, &batIds[i]);
    }
    for (int i = 0; i < FIELDER_COUNT; i++) {
        fieldIds[i] = i;
        pthread_create(&fieldThreads[i], NULL, fielderFunc, &fieldIds[i]);
    }
    
    pthread_join(umpThread, NULL);
    gameOn = false;
    
    for (int i = 0; i < 5; i++) {
        sem_post(&ballReadySem);
        sem_post(&strokeDoneSem);
    }
    pthread_cond_broadcast(&ballHitCond);
    
    for (int i = 0; i < BOWLER_COUNT; i++) pthread_cancel(bowlThreads[i]);
    for (int i = 0; i < BATSMAN_COUNT; i++) pthread_cancel(batThreads[i]);
    for (int i = 0; i < FIELDER_COUNT; i++) pthread_cancel(fieldThreads[i]);
    pthread_cancel(scoreThread);
    pthread_cancel(thirdThread);
    
    usleep(100000);
    
    int tb = game.currOver * 6 + game.currBall;
    allTeams[bt].innScore = game.totalScore;
    allTeams[bt].innWkts = game.wktsDown;
    allTeams[bt].innOvers = game.currOver;
    allTeams[bt].innBalls = game.currBall;
    allTeams[bt].innExtras = game.extrasTotal;
    allTeams[bt].innRR = tb > 0 ? (float)game.totalScore * 6.0 / tb : 0.0;
    
    printf("\n\n");
    printCard();
    printInningSummary(bt);
    
    if (num == 1) game.targetRuns = allTeams[bt].innScore + 1;
}

int main(void) {
    printf("T20 Cricket Simulator\n");
    printf("====================================\n\n");
    srand(time(NULL));
    
    logFile = fopen("match_log.txt", "w");
    if (!logFile) {
        perror("Log file error");
        return 1;
    }
    
    setupTeams();
    setupGame();
    setupSched();
    setupSync();
    setupWFG();
    
    runInnings(1);
    
    printf("\nInnings Break - %s need %d to win\n", allTeams[1].teamName, game.targetRuns);
    
    runInnings(2);
    
    printResult();
    cleanup();
    return 0;
}