# This library provides functionality to interface with a database when performing arc calculations
import arc
import numpy as np
import sqlite3
import pickle
import socket
from datetime import datetime

def convert_c6_calc_options(options):
    # no support for degenerate perturbation yet
    # note theta and phi will be set to 0, since a non zero theta requires degenerate perturbation theory which is not yet supported
    # First, deal with all default options
    if "atom1" not in options:
        options["atom1"] = "na"
    if "atom2" not in options:
        options["atom2"] = "cs"
    if "n1" not in options:
        options["n1"] = 50
    if "n2" not in options:
        options["n2"] = 50
    if "l1" not in options:
        options["l1"] = 0
    if "l2" not in options:
        options["l2"] = 0
    if "j1" not in options:
        options["j1"] = 0.5
    if "j2" not in options:
        options["j2"] = 0.5
#    if "theta" not in options:
#        options["theta"] = 0
#    if "phi" not in options:
#        options["phi"] = 0
    if "dn" not in options:
        options["dn"] = 5
    if "deltaMax" not in options:
        options["deltaMax"] = 10e9 # units of Hz
#    if "degenerate" not in options:
#        options["degenerate"] = False
    res = "c6_" + options["atom1"].lower() + "_" + options["atom2"].lower() + "_n1_" + str(int(options["n1"])) + "_n2_" + str(int(options["n2"])) + "_l1_" + str(int(options["l1"])) + \
          "_l2_" + str(int(options["l2"])) + "_j1_" + "{:.1f}".format(options["j1"]) + "_j2_" +  "{:.1f}".format(options["j2"]) + "_dn_" + str(int(options["dn"])) + "_deltaMax_" + str(int(options["deltaMax"]))
    res = res.replace(".", "_")
    return res

def convert_pair_interaction_options(options):
    # First, deal with all default options
    if "atom1" not in options:
        options["atom1"] = "na"
    if "atom2" not in options:
        options["atom2"] = "cs"
    if "n1" not in options:
        options["n1"] = 50
    if "n2" not in options:
        options["n2"] = 50
    if "l1" not in options:
        options["l1"] = 0
    if "l2" not in options:
        options["l2"] = 0
    if "j1" not in options:
        options["j1"] = 0.5
    if "j2" not in options:
        options["j2"] = 0.5
    if "theta" not in options:
        options["theta"] = 0
    if "phi" not in options:
        options["phi"] = 0
    if "dn" not in options:
        options["dn"] = 5
    if "dl" not in options:
        options["dl"] = 5
    if "deltaMax" not in options:
        options["deltaMax"] = 10e9 # units of Hz
    if "nEig" not in options:
        options["nEig"] = 250
    if "Bz" not in options:
        options["Bz"] = 0
    res = "full_pair_state_" + options["atom1"].lower() + "_" + options["atom2"].lower() + "_n1_" + str(int(options["n1"])) + "_n2_" + str(int(options["n2"])) + "_l1_" + str(int(options["l1"])) + \
            "_l2_" + str(int(options["l2"])) + "_j1_" + "{:.1f}".format(options["j1"]) + "_j2_" +  "{:.1f}".format(options["j2"]) + "_theta_" + "{:.10f}".format(options["theta"]) + "_phi_" + \
            "{:.10f}".format(options["phi"]) + "_dn_" + str(int(options["dn"])) + "_dl_" + str(int(options["dl"])) + "_deltaMax_" + str(int(options["deltaMax"])) + "_nEig_" + \
            str(int(options["nEig"])) + "_Bz_" + "{:.5f}".format(options["Bz"])
    res = res.replace(".", "_")
    return res

def check_valid_atom(name):
    res = (name.lower() == "h") or (name.lower() == "li6") or (name.lower() == "li7") or (name.lower() == "na") or (name.lower() == "k39") or (name.lower() == "k40") \
        or (name.lower() == "k41") or (name.lower() == "rb85") or (name.lower() == "rb87") or (name.lower() == "cs")
    return res

def convert_atom_name_to_atom(name):
    if name.lower() == "h":
        return arc.Hydrogen()
    elif name.lower() == "li6":
        return arc.Lithium6()
    elif name.lower() == "li7":
        return arc.Lithium7()
    elif name.lower() == "na":
        return arc.Sodium()
    elif name.lower() == "k39":
        return arc.Potassium39()
    elif name.lower() == "k40":
        return arc.Potassium40()
    elif name.lower() == "k41":
        return arc.Potassium41()
    elif name.lower() == "rb85":
        return arc.Rubidium85()
    elif name.lower() == "rb87":
        return arc.Rubidium87()
    elif name.lower() == "cs":
        return arc.Cesium()

class Calculation(object):
    def __init__(self, fname):
        self.fname = fname
        self.conn = sqlite3.connect(fname)
        self.cursor = self.conn.cursor()

        # create the metadata table. In this case the basis states table.
        cmd = "CREATE TABLE IF NOT EXISTS pair_interaction_data (tbl_name TEXT PRIMARY KEY, atom1 TEXT NOT NULL, atom2 TEXT NOT NULL, " + \
                "n1 INTEGER NOT NULL, n2 INTEGER NOT NULL, l1 INTEGER NOT NULL, l2 INTEGER NOT NULL, j1_num_states INTEGER NOT NULL, j2_num_states INTEGER NOT NULL, " + \
                "theta REAL NOT NULL, phi REAL NOT NULL, dn INTEGER NOT NULL, dl INTEGER NOT NULL, deltaMaxGHz INTEGER NOT NULL, nEig INTEGER NOT NULL, BzG REAL NOT NULL, states BLOB NOT NULL)"
        self.cursor.execute(cmd)
        self.conn.commit()


    def __del__(self):
        self.conn.close()

    def calculate_c6_perturbatively(self, options):
        # create table if it doesn't exist
        cmd = "CREATE TABLE IF NOT EXISTS c6_table (id TEXT PRIMARY KEY, atom1 TEXT NOT NULL, atom2 TEXT NOT NULL, " + \
                "n1 INTEGER NOT NULL, n2 INTEGER NOT NULL, l1 INTEGER NOT NULL, l2 INTEGER NOT NULL, j1_num_states INTEGER NOT NULL, j2_num_states INTEGER NOT NULL, " + \
                "dn INTEGER NOT NULL, deltaMaxGHz INTEGER NOT NULL, greatest_contributing_state BLOB NOT NULL, " + \
                "defectGHz REAL NOT NULL, C6GHz_um6 REAL NOT NULL, calc_time TEXT NOT NULL, calc_host TEXT NOT NULL)"
        self.cursor.execute(cmd)

        this_id = convert_c6_calc_options(options)

        query = f"SELECT id, greatest_contributing_state, defectGHz, C6GHz_um6 FROM c6_table WHERE id = ? LIMIT 1;"
        self.cursor.execute(query, (this_id,))

        result = self.cursor.fetchone()

        if not result:
            # create entry
            print("Calculating " + str(this_id))
            atom1 = convert_atom_name_to_atom(options["atom1"])
            atom2 = convert_atom_name_to_atom(options["atom2"])
            PI = arc.PairStateInteractions(atom1, options["n1"], options["l1"], options["j1"], options["n2"], options["l2"], options["j2"], 0.5, 0.5, atom2=atom2)
            C6, this_result = PI.getC6perturbatively(0, 0, options["dn"], options["deltaMax"]) # First two arguments are theta and phi
            res_C6 = C6
            res_defect = this_result["defect"]
            res_greatest_contrib_state = [this_result["n1"], this_result["l1"], this_result["j1"], this_result["n2"], this_result["l2"], this_result["j2"]]
            
            hostname = socket.gethostname()
            now = datetime.now()
            time_string = now.strftime("%Y%m%d_%H%M%S")

            query = f"INSERT INTO c6_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            self.cursor.execute(query, (this_id, options["atom1"], options["atom2"], options["n1"], options["n2"], options["l1"], options["l2"], options["j1"] * 2 + 1, \
                                options["j2"] * 2 + 1, options["dn"], options["deltaMax"], pickle.dumps(res_greatest_contrib_state), res_defect, C6, \
                                time_string, hostname))
        else:
            print(str(this_id) + " is already calculated")
            res_greatest_contrib_state = result[1]
            res_defect = result[2]
            res_C6 = result[3]

        self.conn.commit()
        return res_C6, res_greatest_contrib_state, res_defect

    def calculate_pair_interaction(self, rs, options, tolerance=0.01):
        # default tolerance is 0.01 um
        table_name = convert_pair_interaction_options(options)

        # create table if it doesn't exist
        cmd = f"CREATE TABLE IF NOT EXISTS {table_name} (r_um REAL PRIMARY KEY, evalsGHz BLOB NOT NULL, evecs BLOB NOT NULL, calc_time TEXT NOT NULL, calc_host TEXT NOT NULL)"
        #print(cmd)
        self.cursor.execute(cmd)

        # determine which rs have already been calculated
        res_rs = []
        res_evals = []
        res_evecs = []
        res_basis = None
        new_r = []
        for r in rs:
            cmd = f"SELECT * FROM {table_name} WHERE ABS(r_um - ?) < ? LIMIT 1;"
            self.cursor.execute(cmd, (r, tolerance))
            result = self.cursor.fetchone()
            if result:
                print(str(r) + " has already been calculated!")
                # grab the entry
                res_rs.append(result[0])
                res_evals.append(pickle.loads(result[1]))
                res_evecs.append(pickle.loads(result[2]))
            else:
                new_r.append(r)

        # perform calculation for remaining rs
        if len(new_r) != 0:
            print("Calculating for " + str(new_r))
            atom1 = convert_atom_name_to_atom(options["atom1"])
            atom2 = convert_atom_name_to_atom(options["atom2"])
            PI = arc.PairStateInteractions(atom1, options["n1"], options["l1"], options["j1"], options["n2"], options["l2"], options["j2"], 0.5, 0.5, atom2=atom2)
            PI.defineBasis(options["theta"], options["phi"], options["dn"], options["dl"], options["deltaMax"], progressOutput=True, Bz = options["Bz"] * 1e-4)
            PI.diagonalise(np.array(new_r), options["nEig"], progressOutput=True)
            hostname = socket.gethostname()
            now = datetime.now()
            time_string = now.strftime("%Y%m%d_%H%M%S")
            query = f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?)"
            for idx in range(len(new_r)):
                this_r = PI.r[idx]
                this_evals = pickle.dumps(PI.y[idx])
                this_evecs = pickle.dumps(PI.egvecs[idx])
                self.cursor.execute(query, (this_r, this_evals, this_evecs, time_string, hostname))
                res_rs.append(this_r)
                res_evals.append(PI.y[idx])
                res_evecs.append(PI.egvecs[idx])

        # Check if entry already exists in metadata pair_interaction_data table
        query = f"SELECT states FROM pair_interaction_data WHERE tbl_name = ? LIMIT 1;"
        self.cursor.execute(query, (table_name,))
        result = self.cursor.fetchone()
        if not result:
            # Create entry 
            res_basis = PI.basisStates
            query = f"INSERT INTO pair_interaction_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            self.cursor.execute(query, (table_name, options["atom1"], options["atom2"], options["n1"], options["n2"], options["l1"], options["l2"], options["j1"] * 2 + 1, \
                                options["j2"] * 2 + 1, options["theta"], options["phi"], options["dn"], options["dl"], options["deltaMax"], options["nEig"], options["Bz"], pickle.dumps(res_basis)))
        else:
            res_basis = pickle.loads(result[0])

        self.conn.commit()
        return res_rs, res_evals, res_evecs, res_basis

    def list_tables(self):
        # Execute a query to get the list of all tables
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        # Fetch all results from the executed query
        tables = self.cursor.fetchall()

        res = []
        for table in tables:
            res.append(table)

        return res

    def get_table_info(self, table_name):
        # Execute the PRAGMA table_info command to get column information
        self.cursor.execute(f"PRAGMA table_info({table_name});")

        # Fetch all results from the executed query
        columns_info = self.cursor.fetchall()
        return columns_info