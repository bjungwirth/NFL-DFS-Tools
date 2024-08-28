import csv
import json
import math
import os
import random
import time, datetime
import numpy as np
import pulp as plp
import multiprocessing as mp
import pandas as pd
import statistics

# import fuzzywuzzy
import itertools
import collections
import re
from scipy.stats import norm, kendalltau, multivariate_normal, gamma, lognorm
from scipy.stats import truncnorm, truncexpon, gamma
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
import sys


@jit(nopython=True)  # nopython mode ensures the function is fully optimized
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2

pos_own_corr = {
    'QBDST1': 0.936, 'DSTDST0': 0.848, 'RBDST1': 0.835, 'WRDST1': 0.659, 'TEDST1': 0.551,
    'RBK1': 0.517, 'QBK0': 0.511, 'DSTRB1': 0.469, 'QBK1': 0.46, 'WRK0': 0.434,
    'RBK0': 0.405, 'WRK1': 0.381, 'TEK0': 0.261, 'QBWR1': 0.232, 'QBRB1': 0.196,
    'DSTK1': 0.171, 'TEK1': 0.151, 'RBRB0': 0.134, 'DSTQB1': 0.109, 'KDST1': 0.099,
    'TERB1': 0.082, 'KK0': 0.079, 'WRRB1': 0.065, 'DSTWR1': 0.061, 'KRB1': 0.057,
    'TERB0': 0.046, 'WRRB0': 0.033, 'QBTE1': 0.017, 'WRTE1': 0.009, 'KQB1': 0.009,
    'KWR1': 0.005, 'DSTDST1': 0, 'RBWR0': -0.001, 'TEWR0': -0.002, 'KK1': -0.009,
    'KWR0': -0.013, 'KDST0': -0.018, 'WRWR0': -0.024, 'KRB0': -0.031, 'RBTE1': -0.035,
    'TEWR1': -0.035, 'TETE0': -0.071, 'DSTTE1': -0.073, 'KTE1': -0.075, 'RBDST0': -0.097,
    'QBRB0': -0.099, 'TEQB1': -0.115, 'RBTE0': -0.129, 'DSTRB0': -0.157, 'RBWR1': -0.159,
    'WRTE0': -0.176, 'WRQB1': -0.209, 'WRDST0': -0.213, 'KQB0': -0.222, 'KTE0': -0.28,
    'RBRB1': -0.287, 'DSTK0': -0.31, 'DSTWR0': -0.343, 'QBWR0': -0.344, 'WRWR1': -0.346,
    'QBDST0': -0.348, 'TEDST0': -0.385, 'DSTTE0': -0.506, 'QBTE0': -0.532, 'RBQB1': -0.786,
    'RBQB0': -0.91, 'TEQB0': -1.096, 'WRQB0': -1.134, 'DSTQB0': -1.367, 'QBQB0': -1.612,
    'TETE1': -1.742, 'QBQB1': -3.946
}

class NFL_Showdown_Simulator:
    config = None
    player_dict = {}
    field_lineups = {}
    stacks_dict = {}
    gen_lineup_list = []
    roster_construction = []
    id_name_dict = {}
    salary = None
    optimal_score = None
    field_size = None
    team_list = []
    num_iterations = None
    site = None
    payout_structure = {}
    use_contest_data = False
    entry_fee = None
    use_lineup_input = None
    matchups = set()
    projection_minimum = 15
    randomness_amount = 100
    min_lineup_salary = 48000
    max_pct_off_optimal = 0.4
    teams_dict = collections.defaultdict(list)  # Initialize teams_dict
    correlation_rules = {}
    # Add this dictionary at the class level

    def __init__(
        self,
        site,
        field_size,
        num_iterations,
        use_contest_data,
        use_lineup_input,
    ):
        self.site = site
        self.use_lineup_input = use_lineup_input
        self.load_config()
        self.load_rules()

        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["projection_path"]),
        )
        self.load_projections(projection_path)

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["player_path"]),
        )
        self.load_player_ids(player_path)
        self.load_team_stacks()
        self.seen_lineups = {}
        self.seen_lineups_ix = {}

        # ownership_path = os.path.join(
        #    os.path.dirname(__file__),
        #    "../{}_data/{}".format(site, self.config["ownership_path"]),
        # )
        # self.load_ownership(ownership_path)

        # boom_bust_path = os.path.join(
        #    os.path.dirname(__file__),
        #    "../{}_data/{}".format(site, self.config["boom_bust_path"]),
        # )
        # self.load_boom_bust(boom_bust_path)

        #       batting_order_path = os.path.join(
        #           os.path.dirname(__file__),
        #            "../{}_data/{}".format(site, self.config["batting_order_path"]),
        #        )
        #        self.load_batting_order(batting_order_path)

        if site == "dk":
            self.salary = 50000
            self.roster_construction = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]

        elif site == "fd":
            self.salary = 60000
            self.roster_construction = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX"]

        self.use_contest_data = use_contest_data
        if use_contest_data:
            contest_path = os.path.join(
                os.path.dirname(__file__),
                "../{}_data/{}".format(site, self.config["contest_structure_path"]),
            )
            self.load_contest_data(contest_path)
            print("Contest payout structure loaded.")
        else:
            self.field_size = int(field_size)
            self.payout_structure = {0: 0.0}
            self.entry_fee = 0

        # self.adjust_default_stdev()
        self.assertPlayerDict()
        self.num_iterations = int(num_iterations)
        self.get_optimal()
        if self.use_lineup_input:
            self.load_lineups_from_file()
        # if self.match_lineup_input_to_field_size or len(self.field_lineups) == 0:
        # self.generate_field_lineups()
        self.load_correlation_rules()

    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    def load_rules(self):
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.min_lineup_salary = int(self.config["min_lineup_salary"])
        self.max_pct_off_optimal = float(self.config["max_pct_off_optimal"])
        self.pct_field_using_stacks = float(self.config["pct_field_using_stacks"])
        self.default_qb_var = float(self.config["default_qb_var"])
        self.default_skillpos_var = float(self.config["default_skillpos_var"])
        self.default_def_var = float(self.config["default_def_var"])
        self.overlap_limit = float(self.config["num_players_vs_def"])
        self.pct_field_double_stacks = float(self.config["pct_field_double_stacks"])
        self.correlation_rules = self.config["custom_correlations"]
        self.allow_def_vs_qb_cpt = self.config["allow_def_vs_qb_cpt"]

    def assertPlayerDict(self):
        for p, s in list(self.player_dict.items()):
            if s["ID"] == 0 or s["ID"] == "" or s["ID"] is None:
                print(
                    s["Name"]
                    + " name mismatch between projections and player ids, excluding from player_dict"
                )
                self.player_dict.pop(p)

    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that
    # a person could realistically land on this lineup. Skeleton here is taken from base `mlb_optimizer.py`
    def get_optimal(self):
        # print(s['Name'],s['ID'])
        # print(self.player_dict)
        # for p in self.player_dict:
        #    print(p,self.player_dict[p]['UniqueKey'], self.player_dict[p]['fieldFpts'], self.player_dict[p]['Salary'])
        problem = plp.LpProblem("NFL", plp.LpMaximize)
        lp_variables = {
            self.player_dict[player]["UniqueKey"]: plp.LpVariable(
                str(self.player_dict[player]["UniqueKey"]),
                cat="Binary",
            )
            for player in self.player_dict
        }

        # set the objective - maximize fpts & set randomness amount from config
        problem += (
            plp.lpSum(
                self.player_dict[player]["fieldFpts"]
                * lp_variables[self.player_dict[player]["UniqueKey"]]
                for player in self.player_dict
            ),
            "Objective",
        )

        # Set the salary constraints
        max_salary = 50000 if self.site == "dk" else 60000
        min_salary = 44000 if self.site == "dk" else 54000
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Salary"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
            )
            <= max_salary,
            "Max Salary",
        )
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Salary"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["UniqueKey"]]
                for (player, pos_str, team) in self.player_dict
            )
            >= min_salary,
            "Min Salary",
        )

        # Need exactly 1 CPT
        captain_tuples = [
            (player, pos_str, team)
            for (player, pos_str, team) in self.player_dict
            if pos_str == "CPT"
        ]
        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[cpt_tuple]["UniqueKey"]]
                for cpt_tuple in captain_tuples
            )
            == 1,
            f"CPT == 1",
        )

        # Need exactly 5 FLEX on DK, 4 on FD
        flex_tuples = [
            (player, pos_str, team)
            for (player, pos_str, team) in self.player_dict
            if pos_str == "FLEX"
        ]

        number_needed = 5 if self.site == "dk" else 4
        problem += (
            plp.lpSum(
                lp_variables[self.player_dict[flex_tuple]["UniqueKey"]]
                for flex_tuple in flex_tuples
            )
            == number_needed,
            f"FLEX == {number_needed}",
        )

        # Max 5 players from one team if dk, 4 if fd
        for teamIdent in self.team_list:
            players_on_team = [
                (player, position, team)
                for (player, position, team) in self.player_dict
                if teamIdent == team
            ]
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[player_tuple]["UniqueKey"]]
                    for player_tuple in players_on_team
                )
                <= number_needed,
                f"Max {number_needed} players from one team {teamIdent}",
            )

        # Can't roster the same player as cpt and flex
        players_grouped_by_name = {}
        for player, pos_str, team in self.player_dict:
            if player in players_grouped_by_name:
                players_grouped_by_name[player].append((player, pos_str, team))
            else:
                players_grouped_by_name[player] = [(player, pos_str, team)]

        for _, tuple_list in players_grouped_by_name.items():
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[player_tuple]["UniqueKey"]]
                    for player_tuple in tuple_list
                )
                <= 1,
                f"No player in both CPT and FLEX {tuple_list}",
            )

        # Crunch!
        try:
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print(
                "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                    len(self.num_lineups), self.num_lineups
                )
            )

        # problem.writeLP("file.lp")

        # Get the lineup and add it to our list
        player_unqiue_keys = [
            player for player in lp_variables if lp_variables[player].varValue != 0
        ]
        players = []
        for key, value in self.player_dict.items():
            if value["UniqueKey"] in player_unqiue_keys:
                players.append(key)

        fpts_proj = sum(self.player_dict[player]["fieldFpts"] for player in players)
        # sal_used = sum(self.player_dict[player]["Salary"] for player in players)

        var_values = [
            value.varValue for value in problem.variables() if value.varValue != 0
        ]
        player_unqiue_keys = [
            player for player in lp_variables if lp_variables[player].varValue != 0
        ]

        # print((players,player_unqiue_keys, fpts_proj, sal_used,var_values))
        # problem.writeLP("file.lp")

        self.optimal_score = float(fpts_proj)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                name_key = "name" if self.site == "dk" else "nickname"
                if self.site == "fd" and row["position"] == "D":
                    name_key = "last name"
                player_name = row[name_key].replace("-", "#").lower().strip()
                # if qb and dst not in position add flex
                team_key = "teamabbrev" if self.site == "dk" else "team"
                team = row[team_key]
                game_info = "game info" if self.site == "dk" else "game"
                match = re.search(pattern="(\w{2,4}@\w{2,4})", string=row[game_info])
                if match:
                    opp = match.groups()[0].split("@")
                    self.matchups.add((opp[0], opp[1]))
                    for m in opp:
                        if m != team:
                            team_opp = m
                    opp = tuple(opp)
                # if not opp:
                #    print(row)
                if self.site == "dk":
                    position_key = "roster position"
                    position = row[position_key]
                    pos_str = str(position)
                    if (player_name, pos_str, team) in self.player_dict:
                        self.player_dict[(player_name, pos_str, team)]["ID"] = str(
                            row["id"]
                        )
                        self.player_dict[(player_name, pos_str, team)]["Team"] = row[
                            team_key
                        ]
                        self.player_dict[(player_name, pos_str, team)]["Opp"] = team_opp
                        self.player_dict[(player_name, pos_str, team)]["Matchup"] = opp
                        self.player_dict[(player_name, pos_str, team)][
                            "UniqueKey"
                        ] = str(row["id"])
                elif self.site == "fd":
                    for position in ["CPT", "FLEX"]:
                        if (player_name, position, team) in self.player_dict:
                            if position == "CPT":
                                self.player_dict[(player_name, position, team)][
                                    "UniqueKey"
                                ] = f'CPT:{row["id"]}'
                            else:
                                self.player_dict[(player_name, position, team)][
                                    "UniqueKey"
                                ] = f'FLEX:{row["id"]}'
                            self.player_dict[(player_name, position, team)]["ID"] = row[
                                "id"
                            ]
                            self.player_dict[(player_name, position, team)][
                                "Team"
                            ] = row[team_key]
                            self.player_dict[(player_name, position, team)][
                                "Opp"
                            ] = team_opp
                            self.player_dict[(player_name, position, team)][
                                "Matchup"
                            ] = opp
                    self.id_name_dict[str(row["id"])] = row[name_key]

    def load_correlation_rules(self):
        if len(self.correlation_rules.keys()) > 0:
            for primary_player in self.correlation_rules.keys():
                # Convert primary_player to the consistent format
                formatted_primary_player = (
                    primary_player.replace("-", "#").lower().strip()
                )
                for (
                    player_name,
                    pos_str,
                    team,
                ), player_data in self.player_dict.items():
                    if formatted_primary_player == player_name:
                        for second_entity, correlation_value in self.correlation_rules[
                            primary_player
                        ].items():
                            # Convert second_entity to the consistent format
                            formatted_second_entity = (
                                second_entity.replace("-", "#").lower().strip()
                            )

                            # Check if the formatted_second_entity is a player name
                            found_second_entity = False
                            for (
                                se_name,
                                se_pos_str,
                                se_team,
                            ), se_data in self.player_dict.items():
                                if formatted_second_entity == se_name:
                                    player_data["Player Correlations"][
                                        formatted_second_entity
                                    ] = correlation_value
                                    se_data["Player Correlations"][
                                        formatted_primary_player
                                    ] = correlation_value
                                    found_second_entity = True
                                    break

                            # If the second_entity is not found as a player, assume it's a position and update 'Correlations'
                            if not found_second_entity:
                                player_data["Correlations"][
                                    second_entity
                                ] = correlation_value

    def load_contest_data(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                if self.field_size is None:
                    self.field_size = int(row["field size"])
                if self.entry_fee is None:
                    self.entry_fee = float(row["entry fee"])
                # multi-position payouts
                if "-" in row["place"]:
                    indices = row["place"].split("-")
                    # print(indices)
                    # have to add 1 to range to get it to generate value for everything
                    for i in range(int(indices[0]), int(indices[1]) + 1):
                        # print(i)
                        # Where I'm from, we 0 index things. Thus, -1 since Payout starts at 1st place
                        if i >= self.field_size:
                            break
                        self.payout_structure[i - 1] = float(
                            row["payout"].split(".")[0].replace(",", "")
                        )
                # single-position payouts
                else:
                    if int(row["place"]) >= self.field_size:
                        break
                    self.payout_structure[int(row["place"]) - 1] = float(
                        row["payout"].split(".")[0].replace(",", "")
                    )
        # print(self.payout_structure)

    def load_correlation_rules(self):
        if len(self.correlation_rules.keys()) > 0:
            for c in self.correlation_rules.keys():
                for k in self.player_dict:
                    if (
                        c.replace("-", "#").lower().strip()
                        in self.player_dict[k].values()
                    ):
                        for v in self.correlation_rules[c].keys():
                            self.player_dict[k]["Correlations"][
                                v
                            ] = self.correlation_rules[c][v]

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#").lower().strip()
                try:
                    fpts = float(row["fpts"])
                except:
                    fpts = 0
                    print(
                        "unable to load player fpts: "
                        + player_name
                        + ", fpts:"
                        + row["fpts"]
                    )
                if "fieldfpts" in row:
                    if row["fieldfpts"] == "":
                        fieldFpts = fpts
                    else:
                        fieldFpts = float(row["fieldfpts"])
                else:
                    fieldFpts = fpts
                if fpts == 0:
                    continue
                position = [pos for pos in row["position"].split("/")]
                position.sort()
                # if qb and dst not in position add flex
                if self.site == "fd":
                    if "D" in position:
                        position = ["DST"]
                pos = position[0]
                if "stddev" in row:
                    if row["stddev"] == "" or float(row["stddev"]) == 0:
                        if position == "QB":
                            stddev = fpts * self.default_qb_var
                        elif position == "DST":
                            stddev = fpts * self.default_def_var
                        else:
                            stddev = fpts * self.default_skillpos_var
                    else:
                        stddev = float(row["stddev"])
                else:
                    if position == "QB":
                        stddev = fpts * self.default_qb_var
                    elif position == "DST":
                        stddev = fpts * self.default_def_var
                    else:
                        stddev = fpts * self.default_skillpos_var
                # check if ceiling exists in row columns
                if "ceiling" in row:
                    if row["ceiling"] == "" or float(row["ceiling"]) == 0:
                        ceil = fpts + stddev
                    else:
                        ceil = float(row["ceiling"])
                else:
                    ceil = fpts + stddev
                if row["salary"]:
                    sal = float(row["salary"].replace(",", ""))
                # Define the new correlation matrix
                correlation_matrix = {
                    "QB": {"QB": 1.00, "RB": 0.10, "WR": 0.36, "TE": 0.35, "K": -0.02, "DST": 0.04, "Opp QB": 0.23, "Opp RB": 0.07, "Opp WR": 0.12, "Opp TE": 0.10, "Opp K": -0.03, "Opp DST": -0.30},
                    "RB": {"QB": 0.10, "RB": 1.00, "WR": 0.06, "TE": 0.03, "K": 0.16, "DST": 0.10, "Opp QB": 0.07, "Opp RB": -0.02, "Opp WR": 0.05, "Opp TE": 0.07, "Opp K": -0.13, "Opp DST": -0.21},
                    "WR": {"QB": 0.36, "RB": 0.06, "WR": 1.00, "TE": 0.03, "K": 0.00, "DST": 0.06, "Opp QB": 0.12, "Opp RB": 0.05, "Opp WR": 0.05, "Opp TE": 0.06, "Opp K": 0.06, "Opp DST": -0.12},
                    "TE": {"QB": 0.35, "RB": 0.03, "WR": 0.03, "TE": 1.00, "K": 0.02, "DST": 0.00, "Opp QB": 0.10, "Opp RB": 0.04, "Opp WR": 0.06, "Opp TE": 0.09, "Opp K": 0.00, "Opp DST": -0.03},
                    "K": {"QB": -0.02, "RB": 0.16, "WR": 0.00, "TE": 0.02, "K": 1.00, "DST": 0.23, "Opp QB": -0.03, "Opp RB": -0.13, "Opp WR": 0.06, "Opp TE": 0.09, "Opp K": -0.04, "Opp DST": -0.32},
                    "DST": {"QB": 0.04, "RB": 0.10, "WR": 0.06, "TE": 0.00, "K": 0.23, "DST": 1.00, "Opp QB": -0.30, "Opp RB": -0.21, "Opp WR": -0.12, "Opp TE": -0.03, "Opp K": -0.32, "Opp DST": -0.13},
                }
                team = row["team"]
                if team == "LA":
                    team = "LAR"
                if self.site == "fd":
                    if team == "JAX":
                        team = "JAC"
                if team not in self.team_list:
                    self.team_list.append(team)
                own = float(row["own%"].replace("%", ""))
                if own == 0:
                    own = 0.1
                if "cptown%" in row:
                    cptOwn = float(row["cptown%"].replace("%", ""))
                    if cptOwn == 0:
                        cptOwn = 0.1
                else:
                    cptOwn = own * 0.5
                sal = int(row["salary"].replace(",", ""))
                pos_str = "FLEX"
                corr = correlation_matrix.get(pos, {})
                player_data = {
                    "Fpts": fpts,
                    "fieldFpts": fieldFpts,
                    "Position": position,
                    "rosterPosition": "FLEX",
                    "Name": player_name,
                    "Team": team,
                    "Opp": "",
                    "ID": "",
                    "Salary": sal,
                    "StdDev": stddev,
                    "Ceiling": ceil,
                    "Ownership": own,
                    "Correlations": corr,
                    "Player Correlations": {},
                    "In Lineup": False,
                }
                # Check if player is in player_dict and get Opp, ID, Opp Pitcher ID and Opp Pitcher Name
                if (player_name, pos_str, team) in self.player_dict:
                    player_data["Opp"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("Opp", "")
                    player_data["ID"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("ID", "")

                self.player_dict[(player_name, pos_str, team)] = player_data
                self.teams_dict[team].append(player_data)
                pos_str = "CPT"
                if self.site == "dk":
                    cpt_sal = 1.5 * sal
                elif self.site == "fd":
                    cpt_sal = sal
                player_data = {
                    "Fpts": 1.5 * fpts,
                    "fieldFpts": 1.5 * fieldFpts,
                    "Position": position,
                    "rosterPosition": "CPT",
                    "Name": player_name,
                    "Team": team,
                    "Opp": "",
                    "ID": "",
                    "Salary": cpt_sal,
                    "StdDev": stddev,
                    "Ceiling": ceil,
                    "Ownership": cptOwn,
                    "Correlations": corr,
                    "Player Correlations": {},
                    "In Lineup": False,
                }
                # Check if player is in player_dict and get Opp, ID, Opp Pitcher ID and Opp Pitcher Name
                if (player_name, pos_str, team) in self.player_dict:
                    player_data["Opp"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("Opp", "")
                    player_data["ID"] = self.player_dict[
                        (player_name, pos_str, team)
                    ].get("ID", "")

                self.player_dict[(player_name, pos_str, team)] = player_data
                # self.teams_dict[team].append(player_data)  # Add player data to their respective team

    def load_team_stacks(self):
        # Initialize a dictionary to hold QB ownership by team
        qb_ownership_by_team = {}

        for p in self.player_dict:
            # Check if player is a QB
            if "QB" in self.player_dict[p]["Position"]:
                # Fetch the team of the QB
                team = self.player_dict[p]["Team"]

                # Convert the ownership percentage string to a float and divide by 100
                own_percentage = float(self.player_dict[p]["Ownership"]) / 100

                # Add the ownership to the accumulated ownership for the team
                if team in qb_ownership_by_team:
                    qb_ownership_by_team[team] += own_percentage
                else:
                    qb_ownership_by_team[team] = own_percentage

        # Now, update the stacks_dict with the QB ownership by team
        for team, own_percentage in qb_ownership_by_team.items():
            self.stacks_dict[team] = own_percentage

    def extract_id(self, cell_value):
        if "(" in cell_value and ")" in cell_value:
            return cell_value.split("(")[1].replace(")", "")
        else:
            return cell_value

    def load_lineups_from_file(self):
        print("loading lineups")
        i = 0
        path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, "tournament_lineups.csv"),
        )
        with open(path) as file:
            reader = pd.read_csv(file)
            lineup = []
            j = 0
            for i, row in reader.iterrows():
                # print(row)
                if i == self.field_size:
                    break
                lineup = [
                    self.extract_id(str(row[j]))
                    for j in range(len(self.roster_construction))
                ]
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                    if l not in ids:
                        print("lineup {} is missing players {}".format(i, l))
                        if l in self.id_name_dict:
                            print(self.id_name_dict[l])
                        error = True
                if len(lineup) < len(self.roster_construction):
                    print("lineup {} is missing players".format(i))
                    continue
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                if self.site == "fd":
                    un_key_lu = []
                    i = 0
                    for l in lineup:
                        ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                        if l not in ids:
                            print("lineup {} is missing players {}".format(i, l))
                            if l in self.id_name_dict:
                                print(self.id_name_dict[l])
                            error = True
                        else:
                            for k in self.player_dict:
                                if self.player_dict[k]["ID"] == l:
                                    if i == 0:
                                        if (
                                            self.player_dict[k]["rosterPosition"]
                                            == "CPT"
                                        ):
                                            un_key_lu.append(
                                                self.player_dict[k]["UniqueKey"]
                                            )
                                        else:
                                            pass
                                    else:
                                        if (
                                            self.player_dict[k]["rosterPosition"]
                                            == "FLEX"
                                        ):
                                            un_key_lu.append(
                                                self.player_dict[k]["UniqueKey"]
                                            )
                                        else:
                                            pass
                        i += 1
                if len(lineup) < len(self.roster_construction):
                    print("lineup {} is missing players".format(i))
                    continue
                lu = lineup if self.site == "dk" else un_key_lu
                if not error:
                    self.field_lineups[j] = {
                        "Lineup": lu,
                        "Wins": 0,
                        "Top1Percent": 0,
                        "ROI": 0,
                        "Cashes": 0,
                        "Type": "input",
                        "Count": 1,
                    }
                    j += 1
        print("loaded {} lineups".format(j))
        # print(self.field_lineups)

    @staticmethod
    def adjust_ownership(cpt_pos, flex_pos, cpt_team, flex_team, cpt_name, flex_name, cpt_own, flex_own, flex_salary):
        # If the captain and flex are the same player (checking name, position, and team)
        if cpt_name == flex_name and cpt_pos == flex_pos and cpt_team == flex_team:
            return 0.0

        same_team = 1 if cpt_team == flex_team else 0
        position_team_code = cpt_pos + flex_pos + str(same_team)
        
        position_code = pos_own_corr.get(position_team_code, 0)
        
        # Convert ownership percentages to decimals
        cpt_own_decimal = cpt_own / 100
        flex_own_decimal = flex_own / 100
        
        flex_own_given_captain = 1 / (1 + math.exp(-(-3.795) - (0.858 * (flex_own_decimal * 10)) - (0.021 * (10 * cpt_own_decimal)) - (0.050 * (flex_salary / 1000)) - position_code))
        
        # Adjust the ownership to ensure the true probability matches expectation over 5 FLEX selections
        adj_flex_own_decimal = 1 - ((1 - flex_own_given_captain) ** (1 / 5))
        
        # Convert back to percentage
        adj_flex_own = adj_flex_own_decimal * 100
        
        return adj_flex_own

    @staticmethod
    def select_player(
        pos,
        in_lineup,
        ownership,
        ids,
        salaries,
        current_salary,
        remaining_salary,
        k,
        rng,
        salary_ceiling=None,
        salary_floor=None,
        def_opp=None,
        teams=None,
    ):
        valid_players = np.nonzero(
            (pos > 0)
            & (in_lineup == 0)
            & (salaries <= remaining_salary)
            & (
                (current_salary + salaries >= salary_floor)
                if salary_floor is not None
                else True
            )
            # & ((teams != def_opp) if def_opp is not None else True)
        )[0]
        if len(valid_players) == 0:
            # common_indices = set(np.where(pos > 0)[0]) & \
            #     set(np.where(in_lineup == 0)[0]) & \
            #     set(np.where(salaries <= remaining_salary)[0]) & \
            #     set(np.where((current_salary + salaries >= salary_floor) if salary_floor is not None else True)[0]) & \
            #     set(np.where((teams != def_opp) if def_opp is not None else True)[0])
            # print(common_indices)
            # print(current_salary, salary_floor, remaining_salary, k, np.where((current_salary + salaries >= salary_floor)), np.where(pos>0), np.where(salaries <= remaining_salary), np.where(in_lineup == 0), np.where(teams != def_opp) if def_opp is not None else True)
            return None, None
        plyr_list = ids[valid_players]
        prob_list = ownership[valid_players] / ownership[valid_players].sum()
        if salary_ceiling:
            boosted_salaries = np.array([salary_boost(s, salary_ceiling) for s in salaries[valid_players]])
            boosted_probabilities = prob_list * boosted_salaries
            boosted_probabilities /= boosted_probabilities.sum()  # normalize to ensure it sums to 1
            choice = rng.choice(plyr_list, p=boosted_probabilities)
        else:
            choice = rng.choice(plyr_list,p=prob_list)
        return np.where(ids == choice)[0], choice

    @staticmethod
    def validate_lineup(
        salary,
        salary_floor,
        salary_ceiling,
        proj,
        optimal_score,
        max_pct_off_optimal,
        player_teams,
    ):
        reasonable_projection = optimal_score - (max_pct_off_optimal * optimal_score)

        if (
            salary_floor <= salary <= salary_ceiling
            and proj >= reasonable_projection
            and len(set(player_teams)) > 1
        ):
            return True
        return False

    @staticmethod
    def generate_lineups(
        lu_num,
        ids,
        in_lineup,
        pos_matrix,
        ownership,
        salary_floor,
        salary_ceiling,
        optimal_score,
        salaries,
        projections,
        max_pct_off_optimal,
        teams,
        opponents,
        overlap_limit,
        matchups,
        new_player_dict,
        num_players_in_roster,
    ):
        rng = np.random.Generator(np.random.PCG64())
        lus = {}
        in_lineup.fill(0)
        max_iterations = 1000  # Set a maximum number of iterations to prevent infinite loops

        for _ in range(max_iterations):
            salary, proj = 0, 0
            lineup, player_teams, lineup_matchups = [], [], []
            def_opp, players_opposing_def, cpt_selected = None, 0, False
            in_lineup.fill(0)
            remaining_salary = salary_ceiling
            adjusted_ownership = ownership.copy()

            for k, pos in enumerate(pos_matrix.T):
                position_constraint = k >= 1 and players_opposing_def < overlap_limit
                choice_idx, choice = NFL_Showdown_Simulator.select_player(
                    pos,
                    in_lineup,
                    adjusted_ownership if k > 0 else ownership,
                    ids,
                    salaries,
                    salary,
                    remaining_salary,
                    k,
                    rng,
                    salary_ceiling if k == num_players_in_roster - 1 else None,
                    salary_floor if k == num_players_in_roster - 1 else None,
                    def_opp if position_constraint else None,
                    teams if position_constraint else None,
                )
                if choice is None:
                    break

                if k == 0:
                    cpt_player_info = new_player_dict[choice]
                    flex_choice_idx = next(
                        (
                            i
                            for i, v in enumerate(new_player_dict.values())
                            if v["Name"] == cpt_player_info["Name"]
                            and v["Team"] == cpt_player_info["Team"]
                            and v["Position"] == cpt_player_info["Position"]
                            and v["rosterPosition"] == "FLEX"
                        ),
                        None,
                    )
                    if flex_choice_idx is not None:
                        in_lineup[flex_choice_idx] = 1
                    def_opp = opponents[choice_idx][0]
                    cpt_selected = True
                    
                    # Adjust ownership for FLEX players based on the selected captain
                    cpt_pos = cpt_player_info['Position'][0]
                    cpt_team = cpt_player_info['Team']
                    cpt_own = cpt_player_info['Ownership']
                    cpt_name = cpt_player_info['Name']

                    for i, player_id in enumerate(ids):
                        if new_player_dict[player_id]['rosterPosition'] == 'FLEX':
                            flex_pos = new_player_dict[player_id]['Position'][0]
                            flex_team = new_player_dict[player_id]['Team']
                            flex_name = new_player_dict[player_id]['Name']
                            flex_own = ownership[i]
                            flex_salary = salaries[i]
                            
                            adj_own = NFL_Showdown_Simulator.adjust_ownership(cpt_pos, flex_pos, cpt_team, flex_team, cpt_name, flex_name, cpt_own, flex_own, flex_salary)
                            adjusted_ownership[i] = adj_own
                            
                if (
                    cpt_selected
                    and "QB" in new_player_dict[choice]["Position"]
                    and "DEF" in [new_player_dict[x]["Position"] for x in lineup]
                ):
                    break

                lineup.append(str(choice))
                in_lineup[choice_idx] = 1
                salary += salaries[choice_idx]
                proj += projections[choice_idx]
                remaining_salary = salary_ceiling - salary

                player_teams.append(teams[choice_idx][0])

                if teams[choice_idx][0] == def_opp:
                    players_opposing_def += 1

            # Check if the lineup is valid and has the correct number of players
            if (
                len(lineup) == num_players_in_roster
                and NFL_Showdown_Simulator.validate_lineup(
                    salary,
                    salary_floor,
                    salary_ceiling,
                    proj,
                    optimal_score,
                    max_pct_off_optimal,
                    player_teams,
                )
            ):
                lus[lu_num] = {
                    "Lineup": lineup,
                    "Wins": 0,
                    "Top1Percent": 0,
                    "ROI": 0,
                    "Cashes": 0,
                    "Type": "generated",
                    "Count": 0
                }
                return lus

        # If we couldn't generate a valid lineup after max_iterations, return None
        return None

    def remap_player_dict(self, player_dict):
        remapped_dict = {}
        for key, value in player_dict.items():
            if "UniqueKey" in value:
                player_id = value["UniqueKey"]
                remapped_dict[player_id] = value
            else:
                raise KeyError(f"Player details for {key} does not contain an 'ID' key")
        return remapped_dict

    def generate_field_lineups(self):
        diff = self.field_size - len(self.field_lineups)
        if diff <= 0:
            print(
                f"supplied lineups >= contest field size. only retrieving the first {self.field_size} lineups"
            )
            return

        print(f"Generating {diff} lineups.")
        player_data = self.extract_player_data()

        # Initialize problem list
        problems = self.initialize_problems_list(diff, player_data)

        # print(problems[0])

        # Handle stacks logic
        # stacks = self.handle_stacks_logic(diff)
        # print(problems)
        # print(self.player_dict)

        start_time = time.time()

        # Parallel processing for generating lineups
        with mp.Pool() as pool:
            output = pool.starmap(self.generate_lineups, problems)
            pool.close()
            pool.join()

        print("pool closed")

        # Update field lineups
        self.update_field_lineups(output, diff)

        end_time = time.time()
        print(f"lineups took {end_time - start_time} seconds")
        print(f"{diff} field lineups successfully generated")

    def extract_player_data(self):
        ids, ownership, salaries, projections, teams, opponents, matchups, positions = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for player_info in self.player_dict.values():
            if "Team" not in player_info:
                print(
                    f"{player_info['Name']} name mismatch between projections and player ids!"
                )
            ids.append(player_info["UniqueKey"])
            ownership.append(player_info["Ownership"])
            salaries.append(player_info["Salary"])
            projections.append(max(0, player_info.get("fieldFpts", 0)))
            teams.append(player_info["Team"])
            opponents.append(player_info["Opp"])
            matchups.append(player_info["Matchup"])
            pos_list = [
                1 if pos in player_info["rosterPosition"] else 0
                for pos in self.roster_construction
            ]
            positions.append(np.array(pos_list))
        return (
            ids,
            ownership,
            salaries,
            projections,
            teams,
            opponents,
            matchups,
            positions,
        )

    def initialize_problems_list(self, diff, player_data):
        (
            ids,
            ownership,
            salaries,
            projections,
            teams,
            opponents,
            matchups,
            positions,
        ) = player_data
        in_lineup = np.zeros(shape=len(ids))
        ownership, salaries, projections, pos_matrix = map(
            np.array, [ownership, salaries, projections, positions]
        )
        teams, opponents, ids = map(np.array, [teams, opponents, ids])
        new_player_dict = self.remap_player_dict(self.player_dict)
        num_players_in_roster = len(self.roster_construction)
        problems = []
        for i in range(diff):
            lu_tuple = (
                i,
                ids,
                in_lineup,
                pos_matrix,
                ownership,
                self.min_lineup_salary,
                self.salary,
                self.optimal_score,
                salaries,
                projections,
                self.max_pct_off_optimal,
                teams,
                opponents,
                self.overlap_limit,
                matchups,
                new_player_dict,
                num_players_in_roster,
            )
            problems.append(lu_tuple)
        # print(self.player_dict.keys())
        return problems

    def handle_stacks_logic(self, diff):
        stacks = np.random.binomial(
            n=1, p=self.pct_field_using_stacks, size=diff
        ).astype(str)
        stack_len = np.random.choice(
            a=[1, 2],
            p=[1 - self.pct_field_double_stacks, self.pct_field_double_stacks],
            size=diff,
        )
        a = list(self.stacks_dict.keys())
        p = np.array(list(self.stacks_dict.values()))
        probs = p / sum(p)
        for i in range(len(stacks)):
            if stacks[i] == "1":
                choice = random.choices(a, weights=probs, k=1)
                stacks[i] = choice[0]
            else:
                stacks[i] = ""
        return stacks

    def update_field_lineups(self, output, diff):
        if len(self.field_lineups) == 0:
            new_keys = list(range(0, self.field_size))
        else:
            new_keys = list(
                range(
                    max(self.field_lineups.keys()) + 1,
                    max(self.field_lineups.keys()) + 1 + diff,
                )
            )

        nk = new_keys[0]
        for i, o in enumerate(output):
            lineup_list = sorted(next(iter(o.values()))["Lineup"])
            lineup_set = frozenset(lineup_list)  # Convert the list to a frozenset
            #check to make sure that the lineup is valid
            if len(lineup_set) != len(set(lineup_set)):
                print("bad lineup", lineup_set)
                continue
            if len(lineup_set) != len(self.roster_construction):
                print("lineup has wrong number of players", lineup_set)
                print(f"original lineup len: {len(lineup_list)}, lineup set len: {len(lineup_set)}, original lineup: {lineup_list}, lineup set: {lineup_set}")
                continue
            # Keeping track of lineup duplication counts
            if lineup_set in self.seen_lineups:
                self.seen_lineups[lineup_set] += 1

                # Increase the count in field_lineups using the index stored in seen_lineups_ix
                self.field_lineups[self.seen_lineups_ix[lineup_set]]["Count"] += 1
            else:
                self.seen_lineups[lineup_set] = 1

                # Updating the field lineups dictionary
                if nk in self.field_lineups.keys():
                    print("bad lineups dict, please check dk_data files")
                else:
                    self.field_lineups[nk] = next(iter(o.values()))
                    self.field_lineups[nk]['Lineup'] = lineup_set
                    self.field_lineups[nk]['Count'] += self.seen_lineups[lineup_set]     

                    # Store the new nk in seen_lineups_ix for quick access in the future
                    self.seen_lineups_ix[lineup_set] = nk
                    nk += 1

    def calc_gamma(self, mean, sd):
        alpha = (mean / sd) ** 2
        beta = sd**2 / mean
        return alpha, beta

    def run_simulation_for_game(self, team1_id, team1, team2_id, team2, num_iterations):
        def get_corr_value(player1, player2):
            # First, check for specific player-to-player correlations
            if player2["ID"] in player1.get("Player Correlations", {}):
                return player1["Player Correlations"][player2["ID"]]

            # If no specific correlation is found, proceed with the general logic
            position_correlations = {
                "QB": -0.5,
                "RB": -0.05,
                "WR": 0.01,
                "TE": 0.01,
                "K": -0.5,
                "DST": -0.5,
            }

            if player1["Team"] == player2["Team"] and player1["Position"][0] == player2["Position"][0]:
                primary_position = player1["Position"][0]
                return position_correlations[primary_position]

            if player1["Team"] != player2["Team"]:
                player_2_pos = "Opp " + str(player2["Position"][0])
            else:
                player_2_pos = player2["Position"][0]

            return player1["Correlations"].get(
                player_2_pos, 0
            )  # Default to 0 if no correlation is found
        
        def generate_samples(player, num_iterations):
            position = player['Position'][0]
            projected_mean = player['Fpts']
            projected_std = player['StdDev']

            # Set the upper limit as a multiple of the standard deviation above the mean
            upper_limit = projected_mean + 5 * projected_std

            if position in ['WR', 'RB', 'TE']:
                # Use truncated exponential distribution for skill positions
                scale = projected_std
                samples = truncexpon.rvs(b=upper_limit/scale, loc=0, scale=scale, size=num_iterations)
                
            elif position in ['QB', 'DST']:
                # Use truncated normal distribution for QBs and DSTs
                a, b = (0 - projected_mean) / projected_std, (upper_limit - projected_mean) / projected_std
                samples = truncnorm.rvs(a, b, loc=projected_mean, scale=projected_std, size=num_iterations)
            
            elif position == 'K':
                # Use gamma distribution for Kickers, but truncate at upper_limit
                shape = (projected_mean / projected_std) ** 2
                scale = projected_std ** 2 / projected_mean
                samples = gamma.rvs(a=shape, scale=scale, size=num_iterations)
                samples = np.minimum(samples, upper_limit)
            
            else:
                raise ValueError(f"Unknown position: {position}")

            # Ensure non-negative values and match the projected mean exactly
            samples = np.maximum(samples, 0)
            samples = (samples - np.mean(samples)) * (projected_std / np.std(samples)) + projected_mean

            return samples
        
        def build_covariance_matrix(players):
            N = len(players)
            corr_matrix = [[0 for _ in range(N)] for _ in range(N)]

            for i in range(N):
                for j in range(N):
                    if i == j:
                        corr_matrix[i][j] = 1
                    else:
                        corr_matrix[i][j] = get_corr_value(players[i], players[j])
            return np.array(corr_matrix)

        def ensure_positive_definite(matrix):
            # Compute the eigenvalues
            eigenvalues = np.linalg.eigvals(matrix)
            
            # If any eigenvalues are negative or close to zero, adjust them
            if np.any(eigenvalues < 1e-8):
                min_eig = np.min(eigenvalues)
                matrix += (-min_eig + 1e-8) * np.eye(len(matrix))
            
            # Ensure symmetry
            matrix = (matrix + matrix.T) / 2
            
            return matrix

        # Filter out players with projections less than or equal to 0
        team1 = [player for player in team1 if player['Fpts'] > 0]
        team2 = [player for player in team2 if player['Fpts'] > 0]

        game = team1 + team2
        corr_matrix = build_covariance_matrix(game)

        # Ensure the correlation matrix is positive definite
        corr_matrix = ensure_positive_definite(corr_matrix)

        # Generate uncorrelated samples
        uncorrelated_samples = np.array([generate_samples(player, num_iterations) for player in game])

        # Apply correlation
        try:
            L = np.linalg.cholesky(corr_matrix)
            correlated_samples = np.dot(L, uncorrelated_samples)
        except np.linalg.LinAlgError:
            print(f"Warning: Cholesky decomposition failed for {team1_id} vs {team2_id}. Using uncorrelated samples.")
            correlated_samples = uncorrelated_samples

        # Track trimming statistics
        trim_stats = []

        # Ensure means match projected values after correlation
        for i, player in enumerate(game):
            upper_limit = player['Fpts'] + 5 * player['StdDev']
            
            # Count how many samples are above the upper limit
            samples_above_limit = np.sum(correlated_samples[i] > upper_limit)
            
            # Count how many samples are below zero
            samples_below_zero = np.sum(correlated_samples[i] < 0)
            
            # Apply trimming
            correlated_samples[i] = np.minimum(correlated_samples[i], upper_limit)
            correlated_samples[i] = (correlated_samples[i] - np.mean(correlated_samples[i])) * (player['StdDev'] / np.std(correlated_samples[i])) + player['Fpts']
            correlated_samples[i] = np.maximum(correlated_samples[i], 0)  # Ensure non-negative values
            
            # Store trimming statistics
            trim_stats.append({
                'Name': f"{player['Name']} ({player['Team']})",
                'Position': player['Position'][0],
                'Projected Mean': player['Fpts'],
                'Projected StdDev': player['StdDev'],
                'Samples Above Limit': samples_above_limit,
                'Samples Below Zero': samples_below_zero,
                'Percent Above Limit': (samples_above_limit / num_iterations) * 100,
                'Percent Below Zero': (samples_below_zero / num_iterations) * 100
            })

        temp_fpts_dict = {}
        for i, player in enumerate(game):
            temp_fpts_dict[player["UniqueKey"]] = correlated_samples[i]

        # print(f"Starting to generate plots for {team1_id} vs {team2_id}")

        # # Ensure the 'simulation_plots' directory exists
        # os.makedirs('simulation_plots', exist_ok=True)
        # print(f"Created 'simulation_plots' directory")

        # # Plot distributions using KDE
        # plt.figure(figsize=(20, 10))
        # for i, player in enumerate(game):
        #     sns.kdeplot(correlated_samples[i], label=f"{player['Name']} ({player['Team']})")

        # plt.title(f"Fantasy Point Distributions - {team1_id} vs {team2_id}")
        # plt.xlabel("Fantasy Points")
        # plt.ylabel("Density")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        # plt.tight_layout()
        
        # distribution_plot_path = f'simulation_plots/{team1_id}_vs_{team2_id}_distributions.png'
        # plt.savefig(distribution_plot_path)
        # plt.close()
        # print(f"Saved distribution plot to {distribution_plot_path}")

        # # Plot default correlation matrix
        # plt.figure(figsize=(20, 18))
        # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
        #             annot_kws={'size': 8}, fmt='.2f')
        # plt.title("Default Player Correlations")
        
        # # Adjust labels for correlation matrix
        # player_labels = [f"{player['Name']} ({player['Team']})" for player in game]
        # plt.xticks(np.arange(len(player_labels)) + 0.5, player_labels, rotation=90, ha='right', fontsize=8)
        # plt.yticks(np.arange(len(player_labels)) + 0.5, player_labels, rotation=0, fontsize=8)

        # plt.tight_layout()
        # default_corr_plot_path = f'simulation_plots/{team1_id}_vs_{team2_id}_default_correlations.png'
        # plt.savefig(default_corr_plot_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"Saved default correlation plot to {default_corr_plot_path}")

        # # Calculate and plot the correlation matrix of the correlated samples
        # sample_corr_matrix = np.corrcoef(correlated_samples)
        
        # plt.figure(figsize=(20, 18))
        # sns.heatmap(sample_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
        #             annot_kws={'size': 8}, fmt='.2f')
        # plt.title("Sampled Player Correlations")
        
        # # Use the same player labels as before
        # plt.xticks(np.arange(len(player_labels)) + 0.5, player_labels, rotation=90, ha='right', fontsize=8)
        # plt.yticks(np.arange(len(player_labels)) + 0.5, player_labels, rotation=0, fontsize=8)

        # plt.tight_layout()
        # sample_corr_plot_path = f'simulation_plots/{team1_id}_vs_{team2_id}_sampled_correlations.png'
        # plt.savefig(sample_corr_plot_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"Saved sampled correlation plot to {sample_corr_plot_path}")

        # # Create a dataframe with player statistics and trimming info
        # player_stats = []
        # for i, player in enumerate(game):
        #     samples = correlated_samples[i]
        #     stats = {
        #         'Name': f"{player['Name']} ({player['Team']})",
        #         'Position': player['Position'][0],
        #         'Projected Mean': player['Fpts'],
        #         'Projected StdDev': player['StdDev'],
        #         'Sampled Mean': np.mean(samples),
        #         'Sampled StdDev': np.std(samples),
        #         'Sampled Median': np.median(samples),
        #         'Sampled Min': np.min(samples),
        #         'Sampled Max': np.max(samples),
        #         'Percent Above Limit': trim_stats[i]['Percent Above Limit'],
        #         'Percent Below Zero': trim_stats[i]['Percent Below Zero']
        #     }
        #     player_stats.append(stats)

        # stats_df = pd.DataFrame(player_stats)

        # # Plot the statistics table including trimming info
        # plt.figure(figsize=(20, len(game) * 0.5))
        # plt.axis('off')
        # table = plt.table(cellText=stats_df.values,
        #                 colLabels=stats_df.columns,
        #                 cellLoc='center',
        #                 loc='center')
        # table.auto_set_font_size(False)
        # table.set_fontsize(8)
        # table.scale(1, 1.5)
        # plt.title("Player Statistics and Trimming Information", fontsize=16)
        
        # stats_plot_path = f'simulation_plots/{team1_id}_vs_{team2_id}_player_stats_and_trimming.png'
        # plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"Saved player statistics and trimming plot to {stats_plot_path}")

        return temp_fpts_dict

    @staticmethod
    @jit(nopython=True)
    def calculate_payouts(args):
        (
            ranks,
            payout_array,
            entry_fee,
            field_lineup_keys,
            use_contest_data,
            field_lineups_count,
        ) = args
        num_lineups = len(field_lineup_keys)
        combined_result_array = np.zeros(num_lineups)

        payout_cumsum = np.cumsum(payout_array)

        for r in range(ranks.shape[1]):
            ranks_in_sim = ranks[:, r]
            payout_index = 0
            for lineup_index in ranks_in_sim:
                lineup_count = field_lineups_count[lineup_index]
                prize_for_lineup = (
                    (
                        payout_cumsum[payout_index + lineup_count - 1]
                        - payout_cumsum[payout_index - 1]
                    )
                    / lineup_count
                    if payout_index != 0
                    else payout_cumsum[payout_index + lineup_count - 1] / lineup_count
                )
                combined_result_array[lineup_index] += prize_for_lineup
                payout_index += lineup_count
        return combined_result_array

    def run_tournament_simulation(self):
        print(f"Running {self.num_iterations} simulations")
        print(f"Number of unique field lineups: {len(self.field_lineups.keys())}")

        def generate_cpt_outcomes(flex_dict):
            cpt_dict = {}
            for player_id, flex_outcomes in flex_dict.items():
                # Fetch team information using the player_id
                # Assuming self.player_dict uses a structure like {(player_name, position, team): player_data}
                player_data_flex = [
                    data
                    for (name, pos, team), data in self.player_dict.items()
                    if data["UniqueKey"] == player_id and pos == "FLEX"
                ]
                if player_data_flex:
                    player_data_flex = player_data_flex[
                        0
                    ]  # Get the first match (there should only be one)
                    team = player_data_flex["Team"]

                    # Fetch the CPT data using the player_name and team fetched from the above step
                    player_data_cpt = self.player_dict.get(
                        (player_data_flex["Name"], "CPT", team)
                    )
                    if player_data_cpt:
                        cpt_outcomes = flex_outcomes * 1.5
                        cpt_dict[player_data_cpt["UniqueKey"]] = cpt_outcomes
            return cpt_dict

        # # Validation on lineups
        # for f in self.field_lineups:
        #     if len(self.field_lineups[f]["Lineup"]) != len(
        #         self.roster_construction
        #     ):
        #         print("bad lineup", f, self.field_lineups[f])
        #         for p in self.player_dict:
        #             if self.player_dict[p]['UniqueKey'] in self.field_lineups[f]['Lineup']:
        #                 print(self.player_dict[p]['Name'], self.player_dict[p]['Position'], self.player_dict[p]['Team'])

        start_time = time.time()
        temp_fpts_dict = {}

        # Get the only matchup since it's a showdown
        matchup = list(self.matchups)[0]

        # Prepare the arguments for the simulation function
        game_simulation_params = (
            matchup[0],
            self.teams_dict[matchup[0]],
            matchup[1],
            self.teams_dict[matchup[1]],
            self.num_iterations,
        )

        # Run the simulation for the single game
        temp_fpts_dict.update(self.run_simulation_for_game(*game_simulation_params))
        cpt_outcomes_dict = generate_cpt_outcomes(temp_fpts_dict)
        temp_fpts_dict.update(cpt_outcomes_dict)
        # generate arrays for every sim result for each player in the lineup and sum
        fpts_array = np.zeros(shape=(len(self.field_lineups), self.num_iterations))
        # converting payout structure into an np friendly format, could probably just do this in the load contest function
        # print(self.field_lineups)
        # print(temp_fpts_dict)
        # print(payout_array)
        # print(self.player_dict[('patrick mahomes', 'FLEX', 'KC')])
        field_lineups_count = np.array(
            [self.field_lineups[idx]["Count"] for idx in self.field_lineups.keys()]
        )

        for index, values in self.field_lineups.items():
            try:
                fpts_sim = sum(
                    [temp_fpts_dict[player] for player in values["Lineup"]]
                )
            except KeyError:
                for player in values["Lineup"]:
                    if player not in temp_fpts_dict.keys():
                        print(player)
                        # for k,v in self.player_dict.items():
                        # if v['ID'] == player:
                        #        print(k,v)
                # print('cant find player in sim dict', values["Lineup"], temp_fpts_dict.keys())
            # store lineup fpts sum in 2d np array where index (row) corresponds to index of field_lineups and columns are the fpts from each sim
            fpts_array[index] = fpts_sim

        fpts_array = fpts_array.astype(np.float16)
        # ranks = np.argsort(fpts_array, axis=0)[::-1].astype(np.uint16)
        ranks = np.argsort(-fpts_array, axis=0).astype(np.uint32)

        # count wins, top 10s vectorized
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)
        t10, t10_counts = np.unique(ranks[0:9], return_counts=True)
        payout_array = np.array(list(self.payout_structure.values()))
        # subtract entry fee
        payout_array = payout_array - self.entry_fee
        l_array = np.full(
            shape=self.field_size - len(payout_array), fill_value=-self.entry_fee
        )
        payout_array = np.concatenate((payout_array, l_array))
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        chunk_size = self.num_iterations // 16  # Adjust chunk size as needed
        simulation_chunks = [
            (
                ranks[:, i : min(i + chunk_size, self.num_iterations)].copy(),
                payout_array,
                self.entry_fee,
                field_lineups_keys_array,
                self.use_contest_data,
                field_lineups_count,
            )
            for i in range(0, self.num_iterations, chunk_size)
        ]

        # Use the pool to process the chunks in parallel
        with mp.Pool() as pool:
            results = pool.map(self.calculate_payouts, simulation_chunks)

        combined_result_array = np.sum(results, axis=0)
        total_sum = 0
        index_to_key = list(self.field_lineups.keys())
        for idx, roi in enumerate(combined_result_array):
            lineup_key = index_to_key[idx]
            lineup_count = self.field_lineups[lineup_key]["Count"]
            total_sum += roi * lineup_count
            self.field_lineups[lineup_key]["ROI"] += roi

        for idx in self.field_lineups.keys():
            if idx in wins:
                self.field_lineups[idx]["Wins"] += win_counts[
                    np.where(wins == idx)
                ][0]
            if idx in t10:
                self.field_lineups[idx]["Top1Percent"] += t10_counts[
                    np.where(t10 == idx)
                ][0]

        end_time = time.time()
        diff = end_time - start_time
        print(
            str(self.num_iterations)
            + " tournament simulations finished in "
            + str(diff)
            + " seconds. Outputting."
        )

    def output(self):
        unique = {}
        for index, data in self.field_lineups.items():
            # if index == 0:
            #    print(data)
            lineup = data["Lineup"]
            lineup_data = data
            lu_type = lineup_data["Type"]

            salary = 0
            fpts_p = 0
            fieldFpts_p = 0
            ceil_p = 0
            own_p = []
            own_s = []
            lu_names = []
            lu_teams = []
            cpt_tm = ""
            def_opps = []
            players_vs_def = 0

            player_dict_values = {
                v["UniqueKey"]: v for k, v in self.player_dict.items()
            }
            cpt_player = None
            flex_players = []
            lu_names = []
            for player_id in lineup:
                player_data = player_dict_values.get(player_id, {})
                if player_data:
                    if "DST" in player_data["Position"]:
                        def_opps.append(player_data["Opp"])
                    if "CPT" in player_data["rosterPosition"]:
                        cpt_tm = player_data["Team"]
                        cpt_player = player_id
                    else:
                        flex_players.append(player_id)
                    salary += player_data.get("Salary", 0)
                    fpts_p += player_data.get("Fpts", 0)
                    fieldFpts_p += player_data.get("fieldFpts", 0)
                    ceil_p += player_data.get("Ceiling", 0)
                    own_p.append(player_data.get("Ownership", 0) / 100)
                    own_s.append(player_data.get("Ownership", 0))
                    lu_teams.append(player_data["Team"])
                    if "DST" not in player_data["Position"]:
                        if player_data["Team"] in def_opps:
                            players_vs_def += 1
            
            # Sort FLEX players based on their salary
            flex_players.sort(key=lambda pid: player_dict_values[pid]["Salary"], reverse=True)
            
            # Reorder lineup with CPT first, then sorted FLEX players
            sorted_lineup = ([cpt_player] if cpt_player else []) + flex_players
            
            # Create lu_names in the correct order
            for player_id in sorted_lineup:
                player_data = player_dict_values[player_id]
                if self.site == "fd" and "CPT" in player_data["rosterPosition"]:
                    player_id_str = player_data.get("ID", "")
                    if player_id_str.endswith("69696969"):
                        player_id_str = player_id_str.replace("69696969", "")
                    lu_names.append(f"{player_data.get('Name', '')} ({player_id_str})")
                else:
                    lu_names.append(
                        f"{player_data.get('Name', '').replace('#','-')} ({player_data.get('ID', '')})"
                    )
            
            lineup = sorted_lineup
            
            counter = collections.Counter(lu_teams)
            stacks = counter.most_common()

            primary_stack = secondary_stack = ""
            for s in stacks:
                if s[0] == cpt_tm:
                    primary_stack = f"{cpt_tm} {s[1]}"
                    stacks.remove(s)
                    break

            if stacks:
                secondary_stack = f"{stacks[0][0]} {stacks[0][1]}"

            own_p = np.prod(own_p)
            own_s = np.sum(own_s)
            win_p = round(lineup_data["Wins"] / self.num_iterations * 100, 2)
            top10_p = round(lineup_data["Top1Percent"] / self.num_iterations * 100, 2)
            cash_p = round(lineup_data["Cashes"] / self.num_iterations * 100, 2)
            num_dupes = data["Count"]
            if self.use_contest_data:
                roi_p = round(
                    lineup_data["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                )
                roi_round = round(lineup_data["ROI"] / self.num_iterations, 2)

            if self.use_contest_data:
                lineup_str = f"{lu_type},{','.join(lu_names)},{salary},{fpts_p},{fieldFpts_p},{ceil_p},{primary_stack},{secondary_stack},{players_vs_def},{win_p}%,{top10_p}%,{cash_p}%,{own_p},{own_s},{roi_p}%,${roi_round},{num_dupes}"
            else:
                lineup_str = f"{lu_type},{','.join(lu_names)},{salary},{fpts_p},{fieldFpts_p},{ceil_p},{primary_stack},{secondary_stack},{players_vs_def},{win_p}%,{top10_p}%,{cash_p}%,{own_p},{own_s},{num_dupes}"
            unique[
                lineup_str
            ] = fpts_p  # Changed data["Fpts"] to fpts_p, which contains the accumulated Fpts

        return unique

    def player_output(self):
        # out_path = os.path.join(self.output_dir, f"{self.slate_id}_{self.sport}_{self.site}_player_output.csv")
        # First output file
        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_sd_sim_player_exposure_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            f.write(
                "Player,Roster Position,Position,Team,Win%,Top10%,Sim. Own%,Proj. Own%,Avg. Return\n"
            )
            unique_players = {}

            for val in self.field_lineups.values():
                lineup_data = val
                counts = val["Count"]
                for player_id in lineup_data["Lineup"]:
                    if player_id not in unique_players:
                        unique_players[player_id] = {
                            "Wins": lineup_data["Wins"],
                            "Top10": lineup_data["Top1Percent"],
                            "In": val["Count"],
                            "ROI": lineup_data["ROI"],
                        }
                    else:
                        unique_players[player_id]["Wins"] += lineup_data["Wins"]
                        unique_players[player_id]["Top10"] += lineup_data["Top1Percent"]
                        unique_players[player_id]["In"] += val["Count"]
                        unique_players[player_id]["ROI"] += lineup_data["ROI"]

            for player_id, data in unique_players.items():
                field_p = round(data["In"] / self.field_size * 100, 2)
                win_p = round(data["Wins"] / self.num_iterations * 100, 2)
                top10_p = round(data["Top10"] / self.num_iterations / 10 * 100, 2)
                roi_p = round(data["ROI"] / data["In"] / self.num_iterations, 2)
                for k, v in self.player_dict.items():
                    if v["UniqueKey"] == player_id:
                        player_info = v
                        break
                proj_own = player_info.get("Ownership", "N/A")
                p_name = player_info.get("Name", "N/A").replace("#", "-")
                sd_position = player_info.get("rosterPosition", ["N/A"])
                position = player_info.get("Position", ["N/A"])[0]
                team = player_info.get("Team", "N/A")

                f.write(
                    f"{p_name},{sd_position},{position},{team},{win_p}%,{top10_p}%,{field_p}%,{proj_own}%,${roi_p}\n"
                )

    def save_results(self):
        unique = self.output()

        # First output file
        # include timetsamp in filename, formatted as readable
        now = datetime.datetime.now().strftime("%a_%I_%M_%S%p").lower()
        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_sd_sim_lineups_{}_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations, now
            ),
        )
        if self.site == "dk":
            if self.use_contest_data:
                with open(out_path, "w") as f:
                    header = "Type,CPT,FLEX,FLEX,FLEX,FLEX,FLEX,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Players vs DST,Win %,Top 10%,Cash %,Proj. Own. Product,Proj. Own. Sum,ROI%,ROI$,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")
            else:
                with open(out_path, "w") as f:
                    header = "Type,CPT,FLEX,FLEX,FLEX,FLEX,FLEX,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Players vs DST,Win %,Top 10%,Cash %,Proj. Own. Product,Proj. Own. Sum,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")
        else:
            if self.use_contest_data:
                with open(out_path, "w") as f:
                    header = "Type,CPT,FLEX,FLEX,FLEX,FLEX,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Players vs DST,Win %,Top 10%,Cash %,Proj. Own. Product,Proj. Own. Sum,ROI,ROI/Entry Fee,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")
            else:
                with open(out_path, "w") as f:
                    header = "Type,CPT,FLEX,FLEX,FLEX,FLEX,Salary,Fpts Proj,Field Fpts Proj,Ceiling,Primary Stack,Secondary Stack,Players vs DST,Win %,Top 10%,Cash %,Proj. Own. Product,Proj. Own. Sum,Num Dupes\n"
                    f.write(header)
                    for lineup_str, fpts in unique.items():
                        f.write(f"{lineup_str}\n")
        self.player_output()

