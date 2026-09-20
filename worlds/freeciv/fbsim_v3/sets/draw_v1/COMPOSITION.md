# Draw v1 — composition

bank 750  tails 300  mirrors 50  continuous 300  natcond 400

## Binary bank 750: family x horizon
T                      90  120  150  180  210  All
family                                            
EX_comparative          9   11   10   12   11   53
EX_government_at        7   10    9   10   10   46
EX_tech_discovered     10    9   10   10   11   50
NB1_threshold           8    9   11   10   10   48
NB1_value_threshold    10    9   11   11   10   51
NB4_drawdown            4    7   11   11   11   44
NB6_event               5    2    0    0    0    7
NEW_civil_war           0    2    4    5    6   17
NW1_war_at              7   10   10   11   11   49
NW2_diplo               8    8    8   11    9   44
NW5_wonder              6    7    5    8    8   34
S3_wars_at_T            3    0    0    0    0    3
S4_gov_change_count    10    8    5    4    5   32
S5_civ_wonders          8    8   10    9    8   43
S6_city_founding        7    8    7    6    6   34
S7_any_destroyed        2    0    0    0    0    2
W1_wonder_race          7    6    7    8    8   36
W2_directed_conquest    9    9   11   10   10   49
W3_capture_k            8    9    9    6    6   38
W4_lose_k               9    9    7    5    5   35
W5_tech_lead            6    6    4    3    4   23
W6_peace_at             7    3    1    0    1   12
All                   150  150  150  150  150  750

## Binary bank: band x horizon
T           90  120  150  180  210  All
band                                   
0.05-0.23   30   30   30   30   30  150
0.23-0.41   30   30   30   30   30  150
0.41-0.59   30   30   30   30   30  150
0.59-0.77   30   30   30   30   30  150
0.77-0.95   30   30   30   30   30  150
All        150  150  150  150  150  750

## Binary bank: world x horizon
T          90  120  150  180  210  All
world                                 
seed7001   18   22   12    6    8   66
seed7003   20   20   24   24   28  116
seed7005   24   24   23   24   17  112
seed7008   20   16   25   25   26  112
seed7010   17   19   17   24   22   99
seed7011   19   20   23   23   24  109
seed7014   22   22   18   15   22   99
seed7022   10    7    8    9    3   37
All       150  150  150  150  150  750

## Tails 300 (q<=.05): family x horizon
T                     90  120  150  180  210  All
family                                           
EX_comparative         4    4    1    0    0    9
EX_government_at       4    4    5    5    4   22
EX_tech_discovered     3    4    5    5    5   22
NB1_value_threshold    2    4    5    4    5   20
NB4_drawdown           3    4    4    4    5   20
NB6_event              3    0    0    0    0    3
NEW_civil_war          3    4    4    5    4   20
NW1_war_at             3    4    3    3    2   15
NW2_diplo              3    4    4    5    5   21
NW4_survival           1    2    5    4    4   16
NW5_wonder             3    4    4    5    5   21
S4_gov_change_count    4    2    0    0    0    6
S5_civ_wonders         3    4    4    5    5   21
S6_city_founding       2    0    0    0    0    2
W1_wonder_race         4    4    4    4    5   21
W2_directed_conquest   3    4    4    5    4   20
W3_capture_k           3    0    0    0    0    3
W4_lose_k              3    0    0    0    0    3
W5_tech_lead           3    4    4    2    2   15
W6_peace_at            3    4    4    4    5   20
All                   60   60   60   60   60  300
tail q quantiles: {0.1: 0.001, 0.5: 0.013, 0.9: 0.048}  q==0 excluded by construction; per world: {'seed7003': 52, 'seed7008': 51, 'seed7005': 38, 'seed7011': 36, 'seed7001': 32, 'seed7014': 32, 'seed7022': 31, 'seed7010': 28}

## Mirrors 50 (q>=.95): family x horizon
T                     90  120  150  180  210  All
family                                           
EX_comparative         1    0    1    0    0    2
EX_government_at       1    1    1    1    0    4
EX_tech_discovered     0    1    0    0    1    2
NB1_threshold          0    1    1    1    1    4
NB1_value_threshold    1    0    0    1    1    3
NB4_drawdown           0    0    1    1    0    2
NB6_event              1    1    1    0    0    3
NW1_war_at             0    1    0    1    1    3
NW2_diplo              0    0    1    0    0    1
NW5_wonder             1    1    0    1    0    3
S3_wars_at_T           1    0    0    0    0    1
S4_gov_change_count    0    1    1    1    1    4
S5_civ_wonders         0    0    0    1    1    2
S6_city_founding       1    0    0    0    1    2
W1_wonder_race         1    0    0    0    1    2
W2_directed_conquest   0    1    1    0    1    3
W3_capture_k           1    1    1    1    0    4
W4_lose_k              1    1    1    1    1    5
All                   10   10   10   10   10   50

## Continuous 300: family x horizon
T                   90  120  150  180  210  All
family                                         
NC14_world_wonders   6    6    6    7    6   31
NC1_value_at_T      12   12   12   12   12   60
NC5_world_captures   6    6    7    6    7   32
P1_civ_conquests     6    6    7    6    6   31
P2_civ_losses        6    6    6    6    7   31
P3_civ_founds        6    6    6    7    6   31
P5_techs_at_T        6    6    6    6    7   31
P6_world_techs       6    6    3    3    2   20
S7_world_razings     6    6    7    7    7   33
All                 60   60   60   60   60  300
median IQR by family: {'NC14_world_wonders': 2.0, 'NC1_value_at_T': 22.0, 'NC5_world_captures': 19.5, 'P1_civ_conquests': 4.0, 'P2_civ_losses': 5.0, 'P3_civ_founds': 6.0, 'P5_techs_at_T': 4.0, 'P6_world_techs': 9.6, 'S7_world_razings': 8.0}

## Natural conditionals 600: block x horizon
T      120  150  180  210  All
block                         
A        9    9    9    9   36
B        9    9    9    9   36
C1      15   15   15   11   56
C2      42   42   42   46  172
D       25   25   25   25  100
All    100  100  100  100  400

## Natcond: effect size (post hoc, all 1000 replays) by block
stratum  null<.03  small  medium  large>=.15  All
block                                            
A              33      3       0           0   36
B              31      4       1           0   36
C1              7     15      17          17   56
C2             46     51      47          28  172
D              57     23       2          18  100
All           174     96      67          63  400

## Natcond: effect size by horizon
stratum  null<.03  small  medium  large>=.15  All
T                                                
120            34     17      23          26  100
150            45     28      17          10  100
180            35     25      18          22  100
210            60     26       9           5  100
All           174     96      67          63  400

## Natcond: question family x block
block                  A   B  C1   C2    D  All
family                                         
EX_comparative         2   4  19   57    0   82
EX_government_at       2   4   0    0    0    6
EX_tech_discovered     1   0   0    0    0    1
NB1_threshold          2   0   0    0    0    2
NB1_value_threshold    2   4  19   58    0   83
NB4_drawdown           1   4  18   57    0   80
NB6_event              1   0   0    0    0    1
NEW_civil_war          1   0   0    0   18   19
NW1_war_at             1   4   0    0   20   25
NW2_diplo              2   4   0    0   20   26
NW5_wonder             4   0   0    0    0    4
S4_gov_change_count    3   0   0    0    0    3
S5_civ_wonders         2   0   0    0    0    2
S6_city_founding       2   0   0    0    0    2
W1_wonder_race         1   5   0    0    0    6
W2_directed_conquest   3   0   0    0   19   22
W3_capture_k           2   0   0    0    0    2
W4_lose_k              1   0   0    0   19   20
W5_tech_lead           1   4   0    0    0    5
W6_peace_at            2   3   0    0    4    9
All                   36  36  56  172  100  400

## Natcond: reveal type x block
block              A   B  C1   C2    D  All
rev_kind                                   
civ_anarchy        1   3   0    0    0    4
civ_captured       1   4   0    0    0    5
civ_cities_fell2   0   0   1    7    0    8
civ_cities_grew4   6   4   0    0    0   10
civ_govchange      4   8   0    0    0   12
civ_lost           5   0   0   34    0   39
civ_lost3          3   0  55  119   14  191
civ_score_leader   1   0   0   12    0   13
civ_techs_K        3   2   0    0    0    5
civ_wonder         2   3   0    0    0    5
pair_ceasefire     3   9   0    0    0   12
pair_contact       1   0   0    0    0    1
pair_peace         2   3   0    0    0    5
pair_war_began     4   0   0    0   86   90
All               36  36  56  172  100  400
sign among |d|>=.08: + 74  - 56 ; controls: no-news 300 single-prompt 300