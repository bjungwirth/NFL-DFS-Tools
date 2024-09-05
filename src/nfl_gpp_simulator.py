import csv
import json
import math
import os
import random
import time
import numpy as np
import pulp as plp
import multiprocessing as mp
import pandas as pd
import statistics
from multiprocessing import Queue
from tqdm import tqdm
# import fuzzywuzzy
import itertools
import collections
import re
from scipy.stats import norm, kendalltau, multivariate_normal, gamma
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from numba import jit
import datetime

@jit(nopython=True)
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2

class NFL_GPP_Simulator:
    config = None
    player_dict = {}
    field_lineups = {}
    stacks_dict = {}
    gen_lineup_list = []
    roster_construction = []
    game_info = {}
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
    seen_lineups = {}
    seen_lineups_ix = {}
    position_map = {
        0: ["DST"],
        1: ["QB"],
        2: ["RB"],
        3: ["RB"],
        4: ["WR"],
        5: ["WR"],
        6: ["WR"],
        7: ["TE"],
        8: ["RB", "WR", "TE"],
    }

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
            self.roster_construction = [
                "QB",
                "RB",
                "RB",
                "WR",
                "WR",
                "WR",
                "TE",
                "FLEX",
                "DST",
            ]
            self.salary = 50000
            self.max_players_per_team = 8
            self.roster_positions = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']

        elif site == "fd":
            self.roster_construction = [
                "QB",
                "RB",
                "RB",
                "WR",
                "WR",
                "WR",
                "TE",
                "FLEX",
                "DST",
            ]
            self.salary = 60000
            self.max_players_per_team = 4
            self.roster_positions = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']

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
        problem = plp.LpProblem("NFL", plp.LpMaximize)
        lp_variables = {
            self.player_dict[(player, pos_str, team)]["ID"]: plp.LpVariable(
                str(self.player_dict[(player, pos_str, team)]["ID"]), cat="Binary"
            )
            for (player, pos_str, team) in self.player_dict
        }

        # set the objective - maximize fpts
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["fieldFpts"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            ),
            "Objective",
        )

        # Set the salary constraints
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Salary"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            )
            <= self.salary
        )

        if self.site == "dk":
            # Need 1 quarterback
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "QB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need at least 2 RBs can have up to 3 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 2
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 3
            )
            # Need at least 3 WRs can have up to 4 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 3
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 4
            )
            # Need at least 1 TE
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 1
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 2
            )
            # Need 1 DEF
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "DST" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Can only roster 9 total players
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                )
                == 9
            )
            # Max 8 per team in case of weird issues with stacking on short slates
            for team in self.team_list:
                problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in self.player_dict
                        if self.player_dict[(player, pos_str, team)]["Team"] == team
                    )
                    <= 8
                )

        elif self.site == "fd":
            # Need at least 1 point guard, can have up to 3 if utilizing G and UTIL slots
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "QB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need at least 2 RBs can have up to 3 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 2
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 3
            )
            # Need at least 3 WRs can have up to 4 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 3
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 4
            )
            # Need at least 1 TE
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 1
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 2
            )
            # Need 1 DEF
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "DST" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Can only roster 9 total players
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                )
                == 9
            )
            # Max 4 per team
            for team in self.team_list:
                problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in self.player_dict
                        if self.player_dict[(player, pos_str, team)]["Team"] == team
                    )
                    <= 4
                )

        # print(f"Problem Name: {problem.name}")
        # print(f"Sense: {problem.sense}")

        # # Print the objective
        # print("\nObjective:")
        # try:
        #     for v, coef in problem.objective.items():
        #         print(f"{coef}*{v.name}", end=' + ')
        # except Exception as e:
        #     print(f"Error while printing objective: {e}")

        # # Print the constraints
        # print("\nConstraints:")
        # for constraint in problem.constraints.values():
        #     try:
        #         # Extract the left-hand side, right-hand side, and the operator
        #         lhs = "".join(f"{coef}*{var.name}" for var, coef in constraint.items())
        #         rhs = constraint.constant
        #         if constraint.sense == 1:
        #             op = ">="
        #         elif constraint.sense == -1:
        #             op = "<="
        #         else:
        #             op = "="
        #         print(f"{lhs} {op} {rhs}")
        #     except Exception as e:
        #         print(f"Error while printing constraint: {e}")

        # # Print the variables
        # print("\nVariables:")
        # try:
        #     for v in problem.variables():
        #         print(f"{v.name}: LowBound={v.lowBound}, UpBound={v.upBound}, Cat={v.cat}")
        # except Exception as e:
        #     print(f"Error while printing variable: {e}")
        # Crunch!
        try:
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print(
                "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                    len(self.num_lineups), self.num_lineups
                )
            )
        except TypeError:
            for p, s in self.player_dict.items():
                if s["ID"] == 0:
                    print(
                        s["Name"] + " name mismatch between projections and player ids"
                    )
                if s["ID"] == "":
                    print(
                        s["Name"] + " name mismatch between projections and player ids"
                    )
                if s["ID"] is None:
                    print(s["Name"])
        score = str(problem.objective)
        for v in problem.variables():
            score = score.replace(v.name, str(v.varValue))

        self.optimal_score = eval(score)

    @staticmethod
    def extract_matchup_time(game_string):
        # Extract the matchup, date, and time
        match = re.match(
            r"(\w{2,4}@\w{2,4}) (\d{2}/\d{2}/\d{4}) (\d{2}:\d{2}[APM]{2} ET)",
            game_string,
        )

        if match:
            matchup, date, time = match.groups()
            # Convert 12-hour time format to 24-hour format
            time_obj = datetime.datetime.strptime(time, "%I:%M%p ET")
            # Convert the date string to datetime.date
            date_obj = datetime.datetime.strptime(date, "%m/%d/%Y").date()
            # Combine date and time to get a full datetime object
            datetime_obj = datetime.datetime.combine(date_obj, time_obj.time())
            return matchup, datetime_obj
        return None

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                name_key = "name" if self.site == "dk" else "nickname"
                player_name = row[name_key].replace("-", "#").lower().strip()
                # some players have 2 positions - will be listed like 'PG/SF' or 'PF/C'
                position = [pos for pos in row["position"].split("/")]
                position.sort()
                if self.site == "fd":
                    if "D" in position:
                        position = ["DST"]
                        player_name = row['last name'].replace("-", "#").lower().strip()
                # if qb and dst not in position add flex
                if "QB" not in position and "DST" not in position:
                    position.append("FLEX")
                team_key = "teamabbrev" if self.site == "dk" else "team"
                team = row[team_key]
                game_info = "game info" if self.site == "dk" else "game"
                game_info_str = row["game info"] if self.site == "dk" else row["game"]
                result = self.extract_matchup_time(game_info_str)
                match = re.search(pattern="(\w{2,4}@\w{2,4})", string=row[game_info])
                if match:
                    opp = match.groups()[0].split("@")
                    self.matchups.add((opp[0], opp[1]))
                    for m in opp:
                        if m != team:
                            team_opp = m
                    opp = tuple(opp)
                if result:
                    matchup, game_time = result
                    self.game_info[opp] = game_time
                # if not opp:
                #    print(row)
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
                self.id_name_dict[str(row["id"])] = row[name_key]

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
                position = [pos for pos in row["position"].split("/")]
                position.sort()
                # if qb and dst not in position add flex
                if self.site == "fd":
                    if "D" in position:
                        position = ["DST"]
                if "QB" not in position and "DST" not in position:
                    position.append("FLEX")
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
                    sal = int(row["salary"].replace(",", ""))
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
                own = float(row["own%"].replace("%", ""))
                if own == 0:
                    own = 0.1
                pos_str = str(position)
                corr = correlation_matrix.get(pos, {})
                player_data = {
                    "Fpts": fpts,
                    "fieldFpts": fieldFpts,
                    "Position": position,
                    "Name": player_name,
                    "Team": team,
                    "Opp": "",
                    "ID": "",
                    "Salary": int(row["salary"].replace(",", "")),
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
                self.teams_dict[team].append(
                    player_data
                )  # Add player data to their respective team

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
        elif ":" in cell_value:
            return cell_value.split(":")[1]
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
                lineup = [self.extract_id(str(row[j])) for j in range(9)]
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                    if l not in ids:
                        print("lineup {} is missing players {}".format(i, l))
                        if l in self.id_name_dict:
                            print(self.id_name_dict[l])
                        error = True
                if len(lineup) < 9:
                    print("lineup {} is missing players".format(i))
                    continue
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                    if l not in ids:
                        print("lineup {} is missing players {}".format(i, l))
                        if l in self.id_name_dict:
                            print(self.id_name_dict[l])
                        error = True
                if len(lineup) < 9:
                    print("lineup {} is missing players".format(i))
                    continue
                if not error:
                    # reshuffle lineup to match temp_roster_construction
                    temp_roster_construction = [
                        "DST",
                        "QB",
                        "RB",
                        "RB",
                        "WR",
                        "WR",
                        "WR",
                        "TE",
                        "FLEX",
                    ]
                    shuffled_lu = []

                    id_to_player_dict = {
                        v["ID"]: v for k, v in self.player_dict.items()
                    }
                    lineup_copy = lineup.copy()
                    position_counts = {
                        "DST": 0,
                        "QB": 0,
                        "RB": 0,
                        "WR": 0,
                        "TE": 0,
                        "FLEX": 0,
                    }
                    z = 0

                    while z < 9:
                        for t in temp_roster_construction:
                            if position_counts[t] < temp_roster_construction.count(t):
                                for l in lineup_copy:
                                    player_info = id_to_player_dict.get(l)
                                    if player_info and t in player_info["Position"]:
                                        shuffled_lu.append(l)
                                        lineup_copy.remove(l)
                                        position_counts[t] += 1
                                        z += 1
                                        if z == 9:
                                            break
                            if z == 9:
                                break
                    lineup_list = sorted(shuffled_lu)           
                    lineup_set = frozenset(lineup_list)

                    # Keeping track of lineup duplication counts
                    if lineup_set in self.seen_lineups:
                        self.seen_lineups[lineup_set] += 1
                    else:
                        # Add to seen_lineups and seen_lineups_ix
                        self.seen_lineups[lineup_set] = 1
                        self.seen_lineups_ix[lineup_set] = j
                        self.field_lineups[j] = {
                            "Lineup": shuffled_lu,
                            "Wins": 0,
                            "Top1Percent": 0,
                            "ROI": 0,
                            "Cashes": 0,
                            "Type": "opto",
                            "Count" : 1
                        }
                        j += 1
        print("loaded {} lineups".format(j))
        #print(len(self.field_lineups))

    @staticmethod
    def select_player(position, ids, in_lineup, pos_matrix, ownership, salaries, projections, remaining_salary, salary_floor, rng, roster_positions, team_counts, max_players_per_team, teams, overlap_limit, opponents, salary_ceiling, num_players_remaining):
        position_index = roster_positions.index(position)
        valid_indices = np.where((pos_matrix[:, position_index] > 0) & (in_lineup == 0) & (salaries <= remaining_salary))[0]

        if position == 'DST':
            # Ensure DST doesn't exceed overlap limit with offensive players
            valid_indices = [index for index in valid_indices if team_counts.get(opponents[index], 0) <= overlap_limit]
        else:
            # Ensure the player's team doesn't exceed max players per team
            valid_indices = [index for index in valid_indices if team_counts[teams[index]] < max_players_per_team]

        if not valid_indices:
            return None

        probabilities = ownership[valid_indices]
        probabilities /= probabilities.sum()

        chosen_index = rng.choice(valid_indices, p=probabilities)
        chosen_id = ids[chosen_index]

        return chosen_id, salaries[chosen_index], projections[chosen_index]

    @staticmethod
    def is_valid_lineup(lineup, salary, projection, salary_floor, salary_ceiling, optimal_score, max_pct_off_optimal, isStack):
        minimum_projection = optimal_score * (1 - max_pct_off_optimal)
        if salary < salary_floor or salary > salary_ceiling:
            return False
        if projection < minimum_projection:
            return False
        if None in lineup.values():
            return False
        return True

    @staticmethod
    def adjust_probabilities(salaries, ownership, salary_ceiling):
        boosted_salaries = np.array([salary_boost(s, salary_ceiling) for s in salaries])
        boosted_probabilities = ownership * boosted_salaries
        boosted_probabilities /= boosted_probabilities.sum()
        return boosted_probabilities

    @staticmethod
    def build_stack(ids, pos_matrix, teams, team_stack, ownership, stack_positions, rng, roster_positions, in_lineup, stack_len):
        team_indices = np.where(teams == team_stack)[0]
        
        # Find QB
        qb_indices = [i for i in team_indices if in_lineup[i] == 0 and pos_matrix[i][roster_positions.index('QB')] > 0]
        if not qb_indices:
            return [], []
        
        qb_index = rng.choice(qb_indices)
        
        # Find pass catchers (WR and TE)
        pass_catcher_indices = [i for i in team_indices if in_lineup[i] == 0 and
                                (any(pos_matrix[i][roster_positions.index(f'WR{j}')] > 0 for j in range(1, 4)) or
                                pos_matrix[i][roster_positions.index('TE')] > 0)]
        
        if len(pass_catcher_indices) < stack_len - 1:
            return [], []
        
        selected_pass_catchers = rng.choice(pass_catcher_indices, size=stack_len-1, replace=False, p=ownership[pass_catcher_indices]/np.sum(ownership[pass_catcher_indices]))
        
        stack_players = [ids[qb_index]] + list(ids[selected_pass_catchers])
        
        # Assign positions
        slotted_positions = ['QB']
        for idx in selected_pass_catchers:
            if any(pos_matrix[idx][roster_positions.index(f'WR{j}')] > 0 for j in range(1, 4)):
                available_wr_positions = [f'WR{j}' for j in range(1, 4) if f'WR{j}' not in slotted_positions]
                slotted_positions.append(rng.choice(available_wr_positions))
            else:
                slotted_positions.append('TE')
        
        return stack_players, slotted_positions
    
    @staticmethod
    def generate_lineups(params):
        
        rng = np.random.default_rng()
        (lu_num, ids, original_in_lineup, pos_matrix, ownership, initial_salary_floor, salary_ceiling, optimal_score, salaries,
        projections, max_pct_off_optimal, teams, opponents, team_stack, stack_len, overlap_limit,
        max_players_per_team, site, roster_positions) = params

        max_retries = 1000
        salary_floor_decrement = initial_salary_floor * 0.01
        min_projection_decrement_factor = 0.05
        current_salary_floor = initial_salary_floor
        current_projection_factor = 1

        for attempt in range(max_retries):
            in_lineup = original_in_lineup.copy()
            lineup = {position: None for position in roster_positions}
            team_counts = {team: 0 for team in set(teams)} 
            total_salary = 0
            total_projection = 0
            num_players_remaining = len(roster_positions)
            isStack = bool(team_stack)

            if attempt % 100 == 0 and attempt != 0:
                current_salary_floor -= salary_floor_decrement
                current_projection_factor -= min_projection_decrement_factor

            # Implementing stack logic
            if team_stack:
                stack_players, slotted_positions = NFL_GPP_Simulator.build_stack(ids, pos_matrix, teams, team_stack, ownership, ['QB', 'WR', 'TE'], rng, roster_positions, in_lineup, stack_len)
                if stack_players:
                    for player_id, pos in zip(stack_players, slotted_positions):
                        idx = np.where(ids == player_id)[0][0]
                        lineup[pos] = player_id
                        total_salary += salaries[idx]
                        total_projection += projections[idx]
                        in_lineup[idx] = 1
                        team_counts[teams[idx]] += 1
                        num_players_remaining -= 1
                else:
                    continue  # Retry if stack fails

            # Fill other positions
            shuffled_positions = list(roster_positions)
            rng.shuffle(shuffled_positions)
            for position in shuffled_positions:
                if not lineup[position]:
                    result = NFL_GPP_Simulator.select_player(position, ids, in_lineup, pos_matrix, ownership, salaries,
                                                            projections, salary_ceiling - total_salary, current_salary_floor, rng,
                                                            roster_positions, team_counts, max_players_per_team, teams, overlap_limit,
                                                            opponents, salary_ceiling, num_players_remaining)
                    if result:
                        player_id, cost, proj = result
                        idx = np.where(ids == player_id)[0][0]
                        lineup[position] = player_id
                        total_salary += cost
                        total_projection += proj
                        in_lineup[idx] = 1
                        team_counts[teams[idx]] += 1
                        num_players_remaining -= 1
                    else:
                        break  # No valid player found, trigger retry

            if all(value is not None for value in lineup.values()) and NFL_GPP_Simulator.is_valid_lineup(
                    lineup, total_salary, total_projection, current_salary_floor, salary_ceiling, optimal_score, current_projection_factor, isStack):
                return {
                    "Lineup": lineup,
                    "Wins": 0,
                    "Top1Percent": 0,
                    "ROI": 0,
                    "Cashes": 0,
                    "Type": "generated_stack" if isStack else "generated_nostack",
                    "Count": 1,
                    "Ceiling": 0,
                    "Projection": total_projection
                }

        return None


    def setup_stacks(self, diff):
        teams = list(self.stacks_dict.keys())
        probabilities = [self.stacks_dict[team] for team in teams]
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        stacks = []
        stack_lens = []

        for _ in range(diff):
            if np.random.rand() < self.pct_field_using_stacks:
                stack_team = np.random.choice(teams, p=probabilities)
                stack_len = np.random.choice([2, 3], p=[1 - self.pct_field_double_stacks, self.pct_field_double_stacks])
                stacks.append(stack_team)
                stack_lens.append(stack_len)
            else:
                stacks.append('')
                stack_lens.append(0)

        return {'team': stacks, 'len': stack_lens}

    def generate_field_lineups(self):
        diff = self.field_size - len(self.field_lineups)
        if diff <= 0:
            print(f"Supplied lineups >= contest field size. Only retrieving the first {self.field_size} lineups")
            return

        print(f"Generating {diff} lineups.")
        
        ids = []
        ownership = []
        salaries = []
        projections = []
        positions = []
        teams = []
        opponents = []

        temp_roster_construction = [
            "DST",
            "QB",
            "RB",
            "RB",
            "WR",
            "WR",
            "WR",
            "TE",
            "FLEX",
        ]

        for k, player in self.player_dict.items():
            if "Team" not in player:
                print(f"{player['Name']} name mismatch between projections and player ids!")
            
            ids.append(player["ID"])
            ownership.append(player["Ownership"])
            salaries.append(player["Salary"])
            projections.append(player["fieldFpts"] if player["fieldFpts"] >= self.projection_minimum else 0)
            teams.append(player["Team"])
            opponents.append(player["Opp"])
            
            pos_list = [1 if pos in player["Position"] else 0 for pos in self.roster_construction]
            positions.append(np.array(pos_list))

        ids = np.array(ids)
        ownership = np.array(ownership)
        salaries = np.array(salaries)
        projections = np.array(projections)
        pos_matrix = np.array(positions)
        teams = np.array(teams)
        opponents = np.array(opponents)

        stack_config = self.setup_stacks(diff)

        problems = [
            (
                i,
                ids,
                np.zeros(len(ids)),  # fresh in_lineup for each problem
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
                stack_config['team'][i],
                stack_config['len'][i],
                self.overlap_limit,
                self.max_players_per_team,
                self.site,
                self.roster_positions,
            )
            for i in range(diff)
        ]

        start_time = time.time()
        
        successful = mp.Value('i', 0)
        failed = mp.Value('i', 0)
        
        def update_progress(result):
            if result is not None:
                with successful.get_lock():
                    successful.value += 1
            else:
                with failed.get_lock():
                    failed.value += 1
        
        with mp.Pool() as pool:
            pbar = tqdm(total=diff, desc="Generating Lineups")
            
            results = []
            for params in problems:
                result = pool.apply_async(self.generate_lineups, (params,), callback=update_progress)
                results.append(result)
            
            while any(not r.ready() for r in results):
                completed = successful.value + failed.value
                pbar.n = completed
                pbar.set_postfix({'Successful': successful.value, 'Failed': failed.value})
                pbar.refresh()
                time.sleep(0.1)
            
            output = [r.get() for r in results if r.get() is not None]
            pool.close()
            pool.join()
            pbar.close()
        
        print("Pool closed")

        self.update_field_lineups(output, diff)
        
        end_time = time.time()
        print(f"Lineups took {end_time - start_time} seconds")
        print(f"{diff} field lineups successfully generated. {len(self.field_lineups)} unique lineups.")
        print(f"{failed.value} lineups failed to generate")

    def get_start_time(self, player_id):
        for _, player in self.player_dict.items():
            if player["ID"] == player_id:
                matchup = player["Matchup"]
                return self.game_info[matchup]
        return None

    def get_player_attribute(self, player_id, attribute):
        for _, player in self.player_dict.items():
            if player["ID"] == player_id:
                return player.get(attribute, None)
        return None

    def is_valid_for_position(self, player_id, position):
        player_positions = self.get_player_attribute(player_id, "Position")
        if player_positions is None:
            return False

        if position in ['RB1', 'RB2']:
            return 'RB' in player_positions
        elif position in ['WR1', 'WR2', 'WR3']:
            return 'WR' in player_positions
        elif position == 'TE':
            return 'TE' in player_positions
        elif position == 'QB':
            return 'QB' in player_positions
        elif position == 'DST':
            return 'DST' in player_positions
        elif position == 'FLEX':
            return any(pos in player_positions for pos in ['RB', 'WR', 'TE'])
        else:
            print(f"Unknown position: {position}")
            return False


    def sort_lineup_by_start_time(self, lineup):
        flex_player = lineup['FLEX']
        flex_player_start_time = self.get_start_time(flex_player)

        # Initialize variables to track the best swap candidate
        latest_start_time = flex_player_start_time
        swap_candidate_position = None

        # Iterate over RB, WR, and TE positions
        for position in ['RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE']:
            current_player = lineup[position]
            current_player_start_time = self.get_start_time(current_player)

            # Update the latest start time and swap candidate position
            if (current_player_start_time and current_player_start_time > latest_start_time and
                self.is_valid_for_position(flex_player, position) and
                self.is_valid_for_position(current_player, 'FLEX')):

                latest_start_time = current_player_start_time
                swap_candidate_position = position

        # Perform the swap if a suitable candidate is found
        if swap_candidate_position is not None:
            lineup['FLEX'], lineup[swap_candidate_position] = lineup[swap_candidate_position], lineup['FLEX']

        return lineup

    def update_field_lineups(self, output, diff):
        if len(self.field_lineups) == 0:
            new_keys = list(range(0, self.field_size))
        else:
            new_keys = list(range(max(self.field_lineups.keys()) + 1, max(self.field_lineups.keys()) + 1 + diff))

        nk = new_keys[0]
        for o in output:
            # Create a frozenset of player IDs to identify unique lineups regardless of order
            lineup_set = frozenset(o["Lineup"].values())
            
            if lineup_set in self.seen_lineups:
                # Increment the count for this lineup
                self.seen_lineups[lineup_set] += 1
                existing_index = self.seen_lineups_ix[lineup_set]
                self.field_lineups[existing_index]["Count"] += 1
            else:
                # This is a new unique lineup
                self.seen_lineups[lineup_set] = 1
                if nk in self.field_lineups.keys():
                    print("bad lineups dict, please check dk_data files")
                else:
                    if self.site == "dk":
                        sorted_lineup = self.sort_lineup_by_start_time(o["Lineup"])
                    else:
                        sorted_lineup = o["Lineup"]

                    self.field_lineups[nk] = o
                    self.field_lineups[nk]["Lineup"] = sorted_lineup
                    self.field_lineups[nk]["Count"] = 1  # Initialize Count to 1
                    self.field_lineups[nk]["ROI"] = 0  # Initialize ROI to 0
                    self.seen_lineups_ix[lineup_set] = nk
                    nk += 1

        print(f"Total unique lineups: {len(self.field_lineups)}")
        print(f"Total lineups including duplicates: {sum(self.seen_lineups.values())}")

    def calc_gamma(self, mean, sd):
        alpha = (mean / sd) ** 2
        beta = sd**2 / mean
        return alpha, beta

    @staticmethod
    def run_simulation_for_game(team1_id, team1, team2_id, team2, num_iterations):
        sim_rng = np.random.default_rng(seed=int(time.time() * 1000000))
        def get_corr_value(player1, player2):
            # First, check for specific player-to-player correlations
            if player2["ID"] in player1.get("Player Correlations", {}):
                return player1["Player Correlations"][player2["ID"]]

            # If no specific correlation is found, proceed with the general logic
            position_correlations = {
                "QB": -0.5,
                "RB": -0.05,
                "WR": 0.01,
                "TE": -0.02,
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

            upper_limit = projected_mean + 5 * projected_std

            if position in ['WR', 'RB', 'TE']:
                scale = projected_std
                samples = sim_rng.exponential(scale=scale, size=num_iterations)
                samples = np.minimum(samples, upper_limit)
                
            elif position == 'DST':
                samples = sim_rng.normal(loc=projected_mean, scale=projected_std, size=num_iterations)
                samples = np.clip(samples, -4, upper_limit)
            
            elif position == 'QB':
                samples = sim_rng.normal(loc=projected_mean, scale=projected_std, size=num_iterations)
                samples = np.clip(samples, -10, upper_limit)
            
            elif position == 'K':
                shape = (projected_mean / projected_std) ** 2
                scale = projected_std ** 2 / projected_mean
                samples = sim_rng.gamma(shape, scale=scale, size=num_iterations)
                samples = np.minimum(samples, upper_limit)
            
            else:
                raise ValueError(f"Unknown position: {position}")

            # Ensure non-negative values (except for DST) and match the projected mean exactly
            samples = (samples - np.mean(samples)) * (projected_std / np.std(samples)) + projected_mean

            return samples
        
        def build_covariance_matrix(players):
            N = len(players)
            corr_matrix = np.eye(N)  # Start with identity matrix (1s on diagonal)

            for i in range(N):
                for j in range(i+1, N):  # Only compute upper triangle
                    corr_value = get_corr_value(players[i], players[j])
                    corr_matrix[i, j] = corr_value
                    corr_matrix[j, i] = corr_value  # Ensure symmetry

            return corr_matrix

        def ensure_positive_definite(matrix):
            # Compute the eigenvalues
            eigenvalues = np.linalg.eigvals(matrix)
            
            # If any eigenvalues are complex or negative, adjust them
            if np.any(np.imag(eigenvalues) != 0) or np.any(eigenvalues < 1e-8):
                # Use the real part of eigenvalues and ensure they're positive
                min_eig = np.min(np.real(eigenvalues))
                adjustment = max(-min_eig + 1e-8, 0)
                matrix += adjustment * np.eye(len(matrix))
            
            # Ensure symmetry
            matrix = (matrix + matrix.T) / 2
            
            return matrix

        # Debug print
        #print(f"Simulating game: {team1_id} vs {team2_id}")
        #print(f"Number of players in team1: {len(team1)}")
        #print(f"Number of players in team2: {len(team2)}")

        # Filter out players with projections less than or equal to 0
        team1 = [player for player in team1 if player['Fpts'] > 0]
        team2 = [player for player in team2 if player['Fpts'] > 0]

        game = team1 + team2
        
        #print("Players in the game:")
        #for player in game:
        #    print(f"Name: {player['Name']}, Team: {player['Team']}, Position: {player['Position']}, Fpts: {player['Fpts']}, StdDev: {player['StdDev']}")

        corr_matrix = build_covariance_matrix(game)

       # print("\nCorrelation Matrix:")
       # np.set_printoptions(precision=3, suppress=True)
       # print(corr_matrix)

        # Check for symmetry
        #if not np.allclose(corr_matrix, corr_matrix.T):
        #    print("Warning: Correlation matrix is not symmetric")

        # Check for positive semi-definiteness
        eigenvalues = np.linalg.eigvals(corr_matrix)
        #print("\nEigenvalues of the correlation matrix:")
        #print(eigenvalues)

        if np.any(eigenvalues < 0):
            print("Warning: Correlation matrix is not positive semi-definite")

        # Ensure the correlation matrix is positive definite
        corr_matrix = ensure_positive_definite(corr_matrix)

        #print("\nAdjusted Correlation Matrix:")
        #print(corr_matrix)

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
        #trim_stats = []

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
            
            # Calculate additional statistics
            final_mean = np.mean(correlated_samples[i])
            final_std = np.std(correlated_samples[i])
            sample_min = np.min(correlated_samples[i])
            sample_max = np.max(correlated_samples[i])
            
        #     # Store trimming statistics
        #     trim_stats.append({
        #         'Name': f"{player['Name']} ({player['Team']})",
        #         'Position': player['Position'][0],
        #         'Projected Mean': player['Fpts'],
        #         'Projected StdDev': player['StdDev'],
        #         'Final Mean': final_mean,
        #         'Final StdDev': final_std,
        #         'Sampled Min': sample_min,
        #         'Sampled Max': sample_max,
        #         'Samples Above Limit': samples_above_limit,
        #         'Samples Below Zero': samples_below_zero,
        #         'Percent Above Limit': (samples_above_limit / num_iterations) * 100,
        #         'Percent Below Zero': (samples_below_zero / num_iterations) * 100
        #     })

        # # Create DataFrame and set display options
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.width', None)
        # pd.set_option('display.float_format', '{:.2f}'.format)
        
        # trimmed_stats = pd.DataFrame(trim_stats)
        # print(trimmed_stats)
        
        # # Reset display options to default
        # pd.reset_option('display.max_rows')
        # pd.reset_option('display.max_columns')
        # pd.reset_option('display.width')
        # pd.reset_option('display.float_format')

        temp_fpts_dict = {}
        for i, player in enumerate(game):
            temp_fpts_dict[player["UniqueKey"]] = correlated_samples[i]

        # # Modify the plotting code
        # print(f"Starting to generate plots for {team1_id} vs {team2_id}")
        # os.makedirs('simulation_plots', exist_ok=True)

        # team_colors = {team1_id: 'purple', team2_id: 'red'}
        # position_styles = {'QB': '-', 'RB': '--', 'WR': '-.', 'TE': ':', 'K': '-', 'DST': '--'}

        # # Sort players by projected points
        # sorted_players = sorted(enumerate(game), key=lambda x: x[1]['Fpts'], reverse=True)

        # # Split players into three groups
        # n = len(sorted_players)
        # groups = [sorted_players[:n//3], sorted_players[n//3:2*n//3], sorted_players[2*n//3:]]

        # fig, axs = plt.subplots(3, 1, figsize=(20, 30))
        # group_names = ['High Projected', 'Medium Projected', 'Low Projected']

        # for ax, group, name in zip(axs, groups, group_names):
        #     for i, player in group:
        #         team = player['Team']
        #         position = player['Position'][0]
        #         name = player['Name']
                
        #         color = team_colors[team]
        #         style = position_styles[position]
                
        #         sns.kdeplot(correlated_samples[i], color=color, linestyle=style, 
        #                     label=f"{name} ({team} {position})", ax=ax, bw_adjust=1.5)

        #     ax.set_title(f"{name} Players")
        #     ax.set_xlabel("Fantasy Points")
        #     ax.set_ylabel("Density")
        #     ax.set_yscale('log')  # Use log scale for y-axis
        #     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        #     ax.grid(True, alpha=0.3)

        # plt.tight_layout()
        # distribution_plot_path = f'simulation_plots/{team1_id}_vs_{team2_id}_distributions.png'
        # plt.savefig(distribution_plot_path, dpi=300, bbox_inches='tight')
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
        print("Running " + str(self.num_iterations) + " simulations")
        for f in self.field_lineups:
            if len(self.field_lineups[f]["Lineup"]) != 9:
                print("bad lineup", f, self.field_lineups[f])
        print(f"Number of unique field lineups: {len(self.field_lineups.keys())}")

        start_time = time.time()
        temp_fpts_dict = {}
        size = self.num_iterations
        game_simulation_params = []
        for m in self.matchups:
            game_simulation_params.append(
                (
                    m[0],
                    self.teams_dict[m[0]],
                    m[1],
                    self.teams_dict[m[1]],
                    self.num_iterations,
                    self.roster_construction,
                )
            )
        with mp.Pool() as pool:
            results = pool.starmap(self.run_simulation_for_game, game_simulation_params)

        for res in results:
            temp_fpts_dict.update(res)

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
                fpts_sim = sum([temp_fpts_dict[player_id] for player_id in values["Lineup"].values()])
            except KeyError:
                for player_id in values["Lineup"].values():
                    if player_id not in temp_fpts_dict.keys():
                        print(player_id)
            fpts_array[index] = fpts_sim

        fpts_array = fpts_array.astype(np.float16)
        # ranks = np.argsort(fpts_array, axis=0)[::-1].astype(np.uint16)
        ranks = np.argsort(-fpts_array, axis=0).astype(np.uint32)

        # count wins, top 10s vectorized
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)
        cashes, cash_counts = np.unique(ranks[0:len(list(self.payout_structure.values()))], return_counts=True)

        top1pct, top1pct_counts = np.unique(
            ranks[0 : math.ceil(0.01 * len(self.field_lineups)), :], return_counts=True
        )

        payout_array = np.array(list(self.payout_structure.values()))
        # subtract entry fee
        payout_array = payout_array - self.entry_fee
        l_array = np.full(
            shape=self.field_size - len(payout_array), fill_value=-self.entry_fee
        )
        payout_array = np.concatenate((payout_array, l_array))

        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        # Adjusted ROI calculation
        # print(field_lineups_count.shape, payout_array.shape, ranks.shape, fpts_array.shape)

        # Split the simulation indices into chunks
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
            )  # Adding field_lineups_count here
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
            self.field_lineups[lineup_key]["ROI"] += roi

        for idx in self.field_lineups.keys():
            if idx in wins:
                self.field_lineups[idx]["Wins"] += win_counts[np.where(wins == idx)][0]
            if idx in top1pct:
                self.field_lineups[idx]["Top1Percent"] += top1pct_counts[np.where(top1pct == idx)][0]
            if idx in cashes:
                self.field_lineups[idx]["Cashes"] += cash_counts[np.where(cashes == idx)][0]
            
            # Normalize ROI, Wins, Top1Percent, and Cashes by the lineup count
            count = self.field_lineups[idx]["Count"]
            self.field_lineups[idx]["ROI"] /= count
            self.field_lineups[idx]["Wins"]
            self.field_lineups[idx]["Top1Percent"]
            self.field_lineups[idx]["Cashes"]

        end_time = time.time()
        diff = end_time - start_time
        print(
            str(self.num_iterations)
            + " tournament simulations finished in "
            + str(diff)
            + "seconds. Outputting."
        )

    def output(self):
        unique = {}
        for index, x in self.field_lineups.items():
            lu_type = x["Type"]
            salary = 0
            fpts_p = 0
            fieldFpts_p = 0
            ceil_p = 0
            own_p = []
            lu_names = []
            lu_teams = []
            qb_stack = 0
            qb_tm = ""
            players_vs_def = 0
            def_opps = []
            simDupes = x['Count']

            for position, player_id in x["Lineup"].items():
                player = next((v for v in self.player_dict.values() if v["ID"] == player_id), None)
                if player:
                    if "DST" in player["Position"]:
                        def_opps.append(player["Opp"])
                    if "QB" in player["Position"]:
                        qb_tm = player["Team"]
                    
                    salary += player["Salary"]
                    fpts_p += player["Fpts"]
                    fieldFpts_p += player["fieldFpts"]
                    ceil_p += player["Ceiling"]
                    own_p.append(player["Ownership"] / 100)
                    lu_names.append(player["Name"])
                    if "DST" not in player["Position"]:
                        lu_teams.append(player["Team"])
                        if player["Team"] in def_opps:
                            players_vs_def += 1

            counter = collections.Counter(lu_teams)
            stacks = counter.most_common()

            # Find the QB team in stacks and set it as primary stack, remove it from stacks and subtract 1 to make sure qb isn't counted
            primaryStack = ""
            for s in stacks:
                if s[0] == qb_tm:
                    primaryStack = f"{qb_tm} {s[1]}"
                    stacks.remove(s)
                    break

            # After removing QB team, the first team in stacks will be the team with most players not in QB stack
            secondaryStack = f"{stacks[0][0]} {stacks[0][1]}" if stacks else ""
            own_p = np.prod(own_p)
            win_p = round(x["Wins"] / self.num_iterations * 100, 2)
            top10_p = round(x["Top1Percent"] / self.num_iterations * 100, 2)
            cash_p = round(x["Cashes"] / self.num_iterations * 100, 2)

            if self.site == "dk":
                if self.use_contest_data:
                    roi_p = round(x["ROI"] / self.entry_fee / self.num_iterations * 100, 2)
                    roi_round = round(x["ROI"] / x['Count'] / self.num_iterations, 2)
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},${},{}%,{}%,{}%,{},${},{},{},{},{},{}".format(
                        lu_names[self.roster_positions.index('QB')].replace("#", "-"), x["Lineup"]['QB'],
                        lu_names[self.roster_positions.index('RB1')].replace("#", "-"), x["Lineup"]['RB1'],
                        lu_names[self.roster_positions.index('RB2')].replace("#", "-"), x["Lineup"]['RB2'],
                        lu_names[self.roster_positions.index('WR1')].replace("#", "-"), x["Lineup"]['WR1'],
                        lu_names[self.roster_positions.index('WR2')].replace("#", "-"), x["Lineup"]['WR2'],
                        lu_names[self.roster_positions.index('WR3')].replace("#", "-"), x["Lineup"]['WR3'],
                        lu_names[self.roster_positions.index('TE')].replace("#", "-"), x["Lineup"]['TE'],
                        lu_names[self.roster_positions.index('FLEX')].replace("#", "-"), x["Lineup"]['FLEX'],
                        lu_names[self.roster_positions.index('DST')].replace("#", "-"), x["Lineup"]['DST'],
                        fpts_p, fieldFpts_p, ceil_p, salary, win_p, top10_p, roi_p, own_p, roi_round,
                        primaryStack, secondaryStack, players_vs_def, lu_type, simDupes
                    )
                else:
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{}%,{}%,{}%,{},{},{},{},{}".format(
                        lu_names[self.roster_positions.index('QB')].replace("#", "-"), x["Lineup"]['QB'],
                        lu_names[self.roster_positions.index('RB1')].replace("#", "-"), x["Lineup"]['RB1'],
                        lu_names[self.roster_positions.index('RB2')].replace("#", "-"), x["Lineup"]['RB2'],
                        lu_names[self.roster_positions.index('WR1')].replace("#", "-"), x["Lineup"]['WR1'],
                        lu_names[self.roster_positions.index('WR2')].replace("#", "-"), x["Lineup"]['WR2'],
                        lu_names[self.roster_positions.index('WR3')].replace("#", "-"), x["Lineup"]['WR3'],
                        lu_names[self.roster_positions.index('TE')].replace("#", "-"), x["Lineup"]['TE'],
                        lu_names[self.roster_positions.index('FLEX')].replace("#", "-"), x["Lineup"]['FLEX'],
                        lu_names[self.roster_positions.index('DST')].replace("#", "-"), x["Lineup"]['DST'],
                        fpts_p, fieldFpts_p, ceil_p, salary, win_p, top10_p, own_p,
                        primaryStack, secondaryStack, players_vs_def, lu_type, simDupes
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    roi_p = round(
                        x["ROI"] / x['Count'] / self.entry_fee / self.num_iterations * 100, 2
                    )
                    roi_round = round(x["ROI"] / x['Count'] / self.num_iterations, 2)
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{}%,{}%,{}%,{},${},{},{},{},{},{}".format(
                        lu_names[self.roster_positions.index('QB')].replace("#", "-"), x["Lineup"]['QB'],
                        lu_names[self.roster_positions.index('RB1')].replace("#", "-"), x["Lineup"]['RB1'],
                        lu_names[self.roster_positions.index('RB2')].replace("#", "-"), x["Lineup"]['RB2'],
                        lu_names[self.roster_positions.index('WR1')].replace("#", "-"), x["Lineup"]['WR1'],
                        lu_names[self.roster_positions.index('WR2')].replace("#", "-"), x["Lineup"]['WR2'],
                        lu_names[self.roster_positions.index('WR3')].replace("#", "-"), x["Lineup"]['WR3'],
                        lu_names[self.roster_positions.index('TE')].replace("#", "-"), x["Lineup"]['TE'],
                        lu_names[self.roster_positions.index('FLEX')].replace("#", "-"), x["Lineup"]['FLEX'],
                        lu_names[self.roster_positions.index('DST')].replace("#", "-"), x["Lineup"]['DST'],
                        fpts_p,
                        fieldFpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        roi_p,
                        own_p,
                        roi_round,
                        primaryStack,
                        secondaryStack,
                        players_vs_def,
                        lu_type,
                        simDupes
                    )
                else:
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{}%,{}%,{},{},{},{},{},{}".format(
                        lu_names[self.roster_positions.index('QB')].replace("#", "-"), x["Lineup"]['QB'],
                        lu_names[self.roster_positions.index('RB1')].replace("#", "-"), x["Lineup"]['RB1'],
                        lu_names[self.roster_positions.index('RB2')].replace("#", "-"), x["Lineup"]['RB2'],
                        lu_names[self.roster_positions.index('WR1')].replace("#", "-"), x["Lineup"]['WR1'],
                        lu_names[self.roster_positions.index('WR2')].replace("#", "-"), x["Lineup"]['WR2'],
                        lu_names[self.roster_positions.index('WR3')].replace("#", "-"), x["Lineup"]['WR3'],
                        lu_names[self.roster_positions.index('TE')].replace("#", "-"), x["Lineup"]['TE'],
                        lu_names[self.roster_positions.index('FLEX')].replace("#", "-"), x["Lineup"]['FLEX'],
                        lu_names[self.roster_positions.index('DST')].replace("#", "-"), x["Lineup"]['DST'],
                        fpts_p,
                        fieldFpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        own_p,
                        primaryStack,
                        secondaryStack,
                        players_vs_def,
                        lu_type,
                        simDupes
                    )
            unique[index] = lineup_str

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_lineups_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            if self.site == "dk":
                if self.use_contest_data:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product,Avg. Return,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )
                else:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Ceiling,Salary,Win %,Top 10%, Proj. Own. Product,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product,Avg. Return,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )
                else:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Ceiling,Salary,Win %,Top 10%,Proj. Own. Product,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )

            for fpts, lineup_str in unique.items():
                f.write("%s\n" % lineup_str)

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_player_exposure_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            f.write(
                "Player,Position,Team,Win%,Top1%,Sim. Own%,Proj. Own%,Avg. Return\n"
            )
            unique_players = {}
            for val in self.field_lineups.values():
                for player in val["Lineup"].values():
                    if player not in unique_players:
                        unique_players[player] = {
                            "Wins": val["Wins"],
                            "Top1Percent": val["Top1Percent"],
                            "In": val['Count'],
                            "ROI": val["ROI"],
                        }
                    else:
                        unique_players[player]["Wins"] = (
                            unique_players[player]["Wins"] + val["Wins"]
                        )
                        unique_players[player]["Top1Percent"] = (
                            unique_players[player]["Top1Percent"] + val["Top1Percent"]
                        )
                        unique_players[player]["In"] = unique_players[player]["In"] + val['Count']
                        unique_players[player]["ROI"] = (
                            unique_players[player]["ROI"] + val["ROI"]
                        )
            top1PercentCount = (0.01) * self.field_size
            for player, data in unique_players.items():
                field_p = round(data["In"] / self.field_size * 100, 2)
                win_p = round(data["Wins"] / self.num_iterations * 100, 2)
                top10_p = round(data["Top1Percent"] / top1PercentCount / self.num_iterations  * 100, 2)
                roi_p = round(data["ROI"] / data["In"] / self.num_iterations, 2)
                for k, v in self.player_dict.items():
                    if player == v["ID"]:
                        proj_own = v["Ownership"]
                        p_name = v["Name"]
                        position = "/".join(v.get("Position"))
                        team = v.get("Team")
                        proj = v["Fpts"]
                        salary = v["Salary"]
                        break
                f.write(
                    "{},{},{},${},{},{}%,{}%,{}%,${}\n".format(
                        p_name.replace("#", "-"),
                        position,
                        team,
                        salary,
                        proj,
                        win_p,
                        top10_p,
                        field_p,
                        proj_own,
                        roi_p,
                    )
                )
