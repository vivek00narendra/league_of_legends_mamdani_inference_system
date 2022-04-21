# Vivekananad Narendra
# 20111689

from tabnanny import verbose
from simpful import *
import matplotlib.pylab as plt
from numpy import linspace, array

# Boundaries for features found from research
all_data = [[[4.5, 6], [1, 1.5], [1.5, 2.5], [4, 8], [0.2, 0.3], [340, 370]],
            [[1.5, 1.7], [0.8, 1.3], [7.5, 10.5], [9, 14], [0.2, 0.4], [340, 363]],
            [[4.2, 6], [1.2, 1.8], [2.8, 4.2], [6, 8], [0.21, 0.23], [346, 378]],
            [[5.1, 7.5], [1, 1.4], [2.1, 3.1], [7, 9], [0.14, 0.18], [375, 404]],
            [[0.83, 1], [0.7, 1.2], [6.2, 11.6], [4, 6], [0.5, 0.9], [270, 279]]]

# INPUT PARAMETERS (Change for testing)
# Role names have values from 0 to 4
role_number = 0
# Example high input values for top lane
user_inputs = [6.1, 1.51, 2.6, 9, 0.4, 372]

role_names = ["Top", "Jungle", "Mid", "Bot", "Support"]
features_names = ["CS/min", "K/D Ratio", "K/D/A Ratio", "Objectives", "Wards/min", "Gold/min"]

# Global variables storing player role and feature ratings
role = None

# Features are as follows: CS/Min, KD Ratio, KDA Ratio, Objectives, Wards/Min, Gold/min
# 3 signifies the most important features, 2 shows the next most important and 1 shows the least important.
feature_weights = [[3, 2, 1, 2, 1, 3],
                   [1, 1, 3, 3, 2, 2],
                   [3, 2, 1, 2, 1, 3],
                   [3, 2, 1, 2, 1, 3],
                   [1, 1, 3, 2, 3, 2]]

 # Define fuzzy sets and linguistic variables
cs_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=all_data[role_number][0][0]), term="Low")
cs_2 = FuzzySet(function=Triangular_MF(a=0, b=all_data[role_number][0][0], c=all_data[role_number][0][1]), term="Medium")
cs_3 = FuzzySet(function=Triangular_MF(a=all_data[role_number][0][0], b=all_data[role_number][0][1], c=10), term="High")

kd_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=all_data[role_number][1][0]), term="Low")
kd_2 = FuzzySet(function=Triangular_MF(a=0, b=all_data[role_number][1][0], c=all_data[role_number][1][1]), term="Medium")
kd_3 = FuzzySet(function=Triangular_MF(a=all_data[role_number][1][0], b=all_data[role_number][1][1], c=10), term="High")

kda_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=all_data[role_number][2][0]), term="Low")
kda_2 = FuzzySet(function=Triangular_MF(a=0, b=all_data[role_number][2][0], c=all_data[role_number][2][1]), term="Medium")
kda_3 = FuzzySet(function=Triangular_MF(a=all_data[role_number][2][0], b=all_data[role_number][2][1], c=30), term="High")

obj_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=all_data[role_number][3][0]), term="Low")
obj_2 = FuzzySet(function=Triangular_MF(a=0, b=all_data[role_number][3][0], c=all_data[role_number][3][1]), term="Medium")
obj_3 = FuzzySet(function=Triangular_MF(a=all_data[role_number][3][0], b=all_data[role_number][3][1], c=30), term="High")

wards_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=all_data[role_number][4][0]), term="Low")
wards_2 = FuzzySet(function=Triangular_MF(a=0, b=all_data[role_number][4][0], c=all_data[role_number][4][1]), term="Medium")
wards_3 = FuzzySet(function=Triangular_MF(a=all_data[role_number][4][0], b=all_data[role_number][4][1], c=1.5), term="High")

gold_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=all_data[role_number][5][0]), term="Low")
gold_2 = FuzzySet(function=Triangular_MF(a=0, b=all_data[role_number][5][0], c=all_data[role_number][5][1]), term="Medium")
gold_3 = FuzzySet(function=Triangular_MF(a=all_data[role_number][5][0], b=all_data[role_number][5][1], c=500), term="High")

# Define output fuzzy sets and linguistic variable
score_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.50), term="Poor")
score_2 = FuzzySet(function=Triangular_MF(a=0, b=0.50, c=0.75), term="Average")
score_3 = FuzzySet(function=Trapezoidal_MF(a=0, b=0.50, c=0.75, d=0.90), term="Good")
score_4 = FuzzySet(function=Trapezoidal_MF(a=0.50, b=0.75, c=0.90, d=0.100), term="Excellent")

# 5 Different fuzzy models depending on the role.
if role_number == 0:

    # FUZZY SYSTEM FOR TOP
    # Create a fuzzy system object
    FS_top = FuzzySystem()
   
    FS_top.add_linguistic_variable("cs", LinguisticVariable([cs_1, cs_2, cs_3], concept=features_names[0], universe_of_discourse=[0,10]))
    FS_top.add_linguistic_variable("kd", LinguisticVariable([kd_1, kd_2, kd_3], concept=features_names[1], universe_of_discourse=[0,10]))
    FS_top.add_linguistic_variable("kda", LinguisticVariable([kda_1, kda_2, kda_3], concept=features_names[2], universe_of_discourse=[0,10]))
    FS_top.add_linguistic_variable("obj", LinguisticVariable([obj_1, obj_2, obj_3], concept=features_names[3], universe_of_discourse=[0,30]))
    FS_top.add_linguistic_variable("wards", LinguisticVariable([wards_1, wards_2, wards_3], concept=features_names[4], universe_of_discourse=[0,1.5]))
    FS_top.add_linguistic_variable("gold", LinguisticVariable([gold_1, gold_2, gold_3], concept=features_names[5], universe_of_discourse=[0,500]))
    FS_top.add_linguistic_variable("score", LinguisticVariable([score_1, score_2, score_3, score_4], universe_of_discourse=[0,1]))

    # Define fuzzy rules for top
    fuzzy_rules_top = ["IF (cs IS Low) AND (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Medium) AND (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Low) AND (gold IS Medium) THEN (score IS Poor)",
                    "IF (cs IS Medium) AND (gold IS Medium) THEN (score IS Good)",
                    "IF (cs IS Medium) AND (gold IS High) THEN (score IS Excellent)",
                    "IF (cs IS High) AND (gold IS Medium) THEN (score IS Excellent)",
                    "IF (cs IS High) AND (gold IS High) THEN (score IS Excellent)",

                    "IF (cs IS Low) OR (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Medium) OR (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Low) OR (gold IS Medium) THEN (score IS Poor)",
                    "IF (cs IS Medium) OR (gold IS Medium) THEN (score IS Average)",
                    "IF (cs IS Medium) OR (gold IS High) THEN (score IS Good)",
                    "IF (cs IS High) OR (gold IS Medium) THEN (score IS Good)",
                    "IF (cs IS High) OR (gold IS High) THEN (score IS Excellent)",

                    "IF (kd IS Low) AND (obj IS Low) THEN (score IS Poor)",
                    "IF (kd IS Medium) AND (obj IS Low) THEN (score IS Average)",
                    "IF (kd IS Low) AND (obj IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) AND (obj IS Medium) THEN (score IS Good)",
                    "IF (kd IS Medium) AND (obj IS High) THEN (score IS Good)",
                    "IF (kd IS High) AND (obj IS Medium) THEN (score IS Good)",
                    "IF (kd IS High) AND (obj IS High) THEN (score IS Excellent)",

                    "IF (kd IS Low) OR (obj IS Low) THEN (score IS Poor)",
                    "IF (kd IS Medium) OR (obj IS Low) THEN (score IS Average)",
                    "IF (kd IS Low) OR (obj IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) OR (obj IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) OR (obj IS High) THEN (score IS Good)",
                    "IF (kd IS High) OR (obj IS Medium) THEN (score IS Good)",
                    "IF (kd IS High) OR (obj IS High) THEN (score IS Good)",

                    "IF (kda IS Low) AND (wards IS Low) THEN (score IS Poor)",
                    "IF (kda IS Medium) AND (wards IS Low) THEN (score IS Average)",
                    "IF (kda IS Low) AND (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) AND (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) AND (wards IS High) THEN (score IS Average)",
                    "IF (kda IS High) AND (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS High) AND (wards IS High) THEN (score IS Good)"
                    
                    "IF (kda IS Low) OR (wards IS Low) THEN (score IS Poor)",
                    "IF (kda IS Medium) OR (wards IS Low) THEN (score IS Average)",
                    "IF (kda IS Low) OR (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) OR (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) OR (wards IS High) THEN (score IS Average)",
                    "IF (kda IS High) OR (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS High) OR (wards IS High) THEN (score IS Good)"]

    FS_top.add_rules(fuzzy_rules_top)

    # Set antecedents values
    FS_top.set_variable("cs", user_inputs[0])
    FS_top.set_variable("kd", user_inputs[1])
    FS_top.set_variable("kda", user_inputs[2])
    FS_top.set_variable("obj", user_inputs[3])
    FS_top.set_variable("wards", user_inputs[4])
    FS_top.set_variable("gold", user_inputs[5])

    # Perform Mamdani inference and print output
    print(FS_top.Mamdani_inference(["score"], ignore_errors=False, ignore_warnings=False, verbose=False))

    # Plotting surface
    xs = []
    ys = []
    zs = []
    DIVs = 20
    for x in linspace(0,10,DIVs):
        for y in linspace(0,500,DIVs):
            FS_top.set_variable("cs", x)
            FS_top.set_variable("gold", y)
            tip = FS_top.inference()['score']
            xs.append(x)
            ys.append(y)
            zs.append(tip)
    xs = array(xs)
    ys = array(ys)
    zs = array(zs)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx, yy = plt.meshgrid(xs,ys)

    ax.plot_trisurf(xs,ys,zs, vmin=0, vmax=25, cmap='gnuplot2')
    ax.set_xlabel("CS/min")
    ax.set_ylabel("Gold/min")
    ax.set_zlabel("Score")
    ax.set_title("Top: CS/min x Gold/min x Score", pad=20)
    ax.set_zlim(0, 1)
    plt.tight_layout()
    plt.show()

elif role_number == 1:

    # FUZZY SYSTEM FOR JUNGLE
    # Create a fuzzy system object
    FS_jungle = FuzzySystem()
    FS_jungle.add_linguistic_variable("cs", LinguisticVariable([cs_1, cs_2, cs_3], concept=features_names[0], universe_of_discourse=[0,10]))
    FS_jungle.add_linguistic_variable("kd", LinguisticVariable([kd_1, kd_2, kd_3], concept=features_names[1], universe_of_discourse=[0,10]))
    FS_jungle.add_linguistic_variable("kda", LinguisticVariable([kda_1, kda_2, kda_3], concept=features_names[2], universe_of_discourse=[0,30]))
    FS_jungle.add_linguistic_variable("obj", LinguisticVariable([obj_1, obj_2, obj_3], concept=features_names[3], universe_of_discourse=[0,30]))
    FS_jungle.add_linguistic_variable("wards", LinguisticVariable([wards_1, wards_2, wards_3], concept=features_names[4], universe_of_discourse=[0,1.5]))
    FS_jungle.add_linguistic_variable("gold", LinguisticVariable([gold_1, gold_2, gold_3], concept=features_names[5], universe_of_discourse=[0,500]))
    FS_jungle.add_linguistic_variable("score", LinguisticVariable([score_1, score_2, score_3, score_4], universe_of_discourse=[0,1]))

    # Define fuzzy rules for jungle
    fuzzy_rules_jungle = ["IF (kda IS Low) AND (obj IS Low) THEN (score IS Poor)",
                    "IF (kda IS Medium) AND (obj IS Low) THEN (score IS Poor)",
                    "IF (kda IS Low) AND (obj IS Medium) THEN (score IS Poor)",
                    "IF (kda IS Medium) AND (obj IS Medium) THEN (score IS Good)",
                    "IF (kda IS Medium) AND (obj IS High) THEN (score IS Excellent)",
                    "IF (kda IS High) AND (obj IS Medium) THEN (score IS Excellent)",
                    "IF (kda IS High) AND (obj IS High) THEN (score IS Excellent)",

                    "IF (kda IS Low) OR (obj IS Low) THEN (score IS Poor)",
                    "IF (kda IS Medium) OR (obj IS Low) THEN (score IS Poor)",
                    "IF (kda IS Low) OR (obj IS Medium) THEN (score IS Poor)",
                    "IF (kda IS Medium) OR (obj IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) OR (obj IS High) THEN (score IS Good)",
                    "IF (kda IS High) OR (obj IS Medium) THEN (score IS Good)",
                    "IF (kda IS High) OR (obj IS High) THEN (score IS Excellent)",

                    "IF (wards IS Low) AND (gold IS Low) THEN (score IS Poor)",
                    "IF (wards IS Medium) AND (gold IS Low) THEN (score IS Average)",
                    "IF (wards IS Low) AND (gold IS Medium) THEN (score IS Average)",
                    "IF (wards IS Medium) AND (gold IS Medium) THEN (score IS Good)",
                    "IF (wards IS Medium) AND (gold IS High) THEN (score IS Good)",
                    "IF (wards IS High) AND (gold IS Medium) THEN (score IS Good)",
                    "IF (wards IS High) AND (gold IS High) THEN (score IS Excellent)",

                    "IF (wards IS Low) OR (gold IS Low) THEN (score IS Poor)",
                    "IF (wards IS Medium) OR (gold IS Low) THEN (score IS Average)",
                    "IF (wards IS Low) OR (gold IS Medium) THEN (score IS Average)",
                    "IF (wards IS Medium) OR (gold IS Medium) THEN (score IS Average)",
                    "IF (wards IS Medium) OR (gold IS High) THEN (score IS Good)",
                    "IF (wards IS High) OR (gold IS Medium) THEN (score IS Good)",
                    "IF (wards IS High) OR (gold IS High) THEN (score IS Good)",

                    "IF (kd IS Low) AND (cs IS Low) THEN (score IS Poor)",
                    "IF (kd IS Medium) AND (cs IS Low) THEN (score IS Average)",
                    "IF (kd IS Low) AND (cs IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) AND (cs IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) AND (cs IS High) THEN (score IS Average)",
                    "IF (kd IS High) AND (cs IS Medium) THEN (score IS Average)",
                    "IF (kd IS High) AND (cs IS High) THEN (score IS Good)"
                    
                    "IF (kd IS Low) OR (cs IS Low) THEN (score IS Poor)",
                    "IF (kd IS Medium) OR (cs IS Low) THEN (score IS Average)",
                    "IF (kd IS Low) OR (cs IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) OR (cs IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) OR (cs IS High) THEN (score IS Average)",
                    "IF (kd IS High) OR (cs IS Medium) THEN (score IS Average)",
                    "IF (kd IS High) OR (cs IS High) THEN (score IS Good)"]

    FS_jungle.add_rules(fuzzy_rules_jungle)

    # Set antecedents values
    FS_jungle.set_variable("cs", user_inputs[0])
    FS_jungle.set_variable("kd", user_inputs[1])
    FS_jungle.set_variable("kda", user_inputs[2])
    FS_jungle.set_variable("obj", user_inputs[3])
    FS_jungle.set_variable("wards", user_inputs[4])
    FS_jungle.set_variable("gold", user_inputs[5])

    # Perform Mamdani inference and print output
    print(FS_jungle.Mamdani_inference(["score"], ignore_errors=False, ignore_warnings=False, verbose=True))

    # Plotting surface
    xs = []
    ys = []
    zs = []
    DIVs = 20
    for x in linspace(0,30,DIVs):
        for y in linspace(0,30,DIVs):
            FS_jungle.set_variable("kda", x)
            FS_jungle.set_variable("obj", y)
            tip = FS_jungle.inference()['score']
            xs.append(x)
            ys.append(y)
            zs.append(tip)
    xs = array(xs)
    ys = array(ys)
    zs = array(zs)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx, yy = plt.meshgrid(xs,ys)

    ax.plot_trisurf(xs,ys,zs, vmin=0, vmax=25, cmap='gnuplot2')
    ax.set_xlabel("K/D/A Ratio")
    ax.set_ylabel("Objectives")
    ax.set_zlabel("Score")
    ax.set_title("Jungle: K/D/A Ratio x Objectives x Score", pad=20)
    ax.set_zlim(0, 1)
    plt.tight_layout()
    plt.show()


elif role_number == 2:

    # FUZZY SYSTEM FOR MID
    # Create a fuzzy system object
    FS_mid = FuzzySystem()

    FS_mid.add_linguistic_variable("cs", LinguisticVariable([cs_1, cs_2, cs_3], concept=features_names[0], universe_of_discourse=[0,10]))
    FS_mid.add_linguistic_variable("kd", LinguisticVariable([kd_1, kd_2, kd_3], concept=features_names[1], universe_of_discourse=[0,10]))
    FS_mid.add_linguistic_variable("kda", LinguisticVariable([kda_1, kda_2, kda_3], concept=features_names[2], universe_of_discourse=[0,30]))
    FS_mid.add_linguistic_variable("obj", LinguisticVariable([obj_1, obj_2, obj_3], concept=features_names[3], universe_of_discourse=[0,30]))
    FS_mid.add_linguistic_variable("wards", LinguisticVariable([wards_1, wards_2, wards_3], concept=features_names[4], universe_of_discourse=[0,1.5]))
    FS_mid.add_linguistic_variable("gold", LinguisticVariable([gold_1, gold_2, gold_3], concept=features_names[5], universe_of_discourse=[0,500]))
    FS_mid.add_linguistic_variable("score", LinguisticVariable([score_1, score_2, score_3, score_4], universe_of_discourse=[0,1]))

    # Define fuzzy rules for mid
    fuzzy_rules_mid = ["IF (cs IS Low) AND (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Medium) AND (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Low) AND (gold IS Medium) THEN (score IS Poor)",
                    "IF (cs IS Medium) AND (gold IS Medium) THEN (score IS Good)",
                    "IF (cs IS Medium) AND (gold IS High) THEN (score IS Excellent)",
                    "IF (cs IS High) AND (gold IS Medium) THEN (score IS Excellent)",
                    "IF (cs IS High) AND (gold IS High) THEN (score IS Excellent)",

                    "IF (cs IS Low) OR (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Medium) OR (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Low) OR (gold IS Medium) THEN (score IS Poor)",
                    "IF (cs IS Medium) OR (gold IS Medium) THEN (score IS Average)",
                    "IF (cs IS Medium) OR (gold IS High) THEN (score IS Good)",
                    "IF (cs IS High) OR (gold IS Medium) THEN (score IS Good)",
                    "IF (cs IS High) OR (gold IS High) THEN (score IS Excellent)",

                    "IF (gold IS Low) AND (obj IS Low) THEN (score IS Poor)",
                    "IF (gold IS Medium) AND (obj IS Low) THEN (score IS Average)",
                    "IF (gold IS Low) AND (obj IS Medium) THEN (score IS Average)",
                    "IF (gold IS Medium) AND (obj IS Medium) THEN (score IS Good)",
                    "IF (gold IS Medium) AND (obj IS High) THEN (score IS Good)",
                    "IF (gold IS High) AND (obj IS Medium) THEN (score IS Good)",
                    "IF (gold IS High) AND (obj IS High) THEN (score IS Excellent)",

                    "IF (gold IS Low) OR (obj IS Low) THEN (score IS Poor)",
                    "IF (gold IS Medium) OR (obj IS Low) THEN (score IS Average)",
                    "IF (gold IS Low) OR (obj IS Medium) THEN (score IS Average)",
                    "IF (gold IS Medium) OR (obj IS Medium) THEN (score IS Average)",
                    "IF (gold IS Medium) OR (obj IS High) THEN (score IS Good)",
                    "IF (gold IS High) OR (obj IS Medium) THEN (score IS Good)",
                    "IF (gold IS High) OR (obj IS High) THEN (score IS Good)",

                    "IF (kda IS Low) AND (wards IS Low) THEN (score IS Poor)",
                    "IF (kda IS Medium) AND (wards IS Low) THEN (score IS Average)",
                    "IF (kda IS Low) AND (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) AND (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) AND (wards IS High) THEN (score IS Average)",
                    "IF (kda IS High) AND (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS High) AND (wards IS High) THEN (score IS Good)"
                    
                    "IF (kda IS Low) OR (wards IS Low) THEN (score IS Poor)",
                    "IF (kda IS Medium) OR (wards IS Low) THEN (score IS Average)",
                    "IF (kda IS Low) OR (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) OR (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) OR (wards IS High) THEN (score IS Average)",
                    "IF (kda IS High) OR (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS High) OR (wards IS High) THEN (score IS Good)"]

    FS_mid.add_rules(fuzzy_rules_mid)

    # Set antecedents values
    FS_mid.set_variable("cs", user_inputs[0])
    FS_mid.set_variable("kd", user_inputs[1])
    FS_mid.set_variable("kda", user_inputs[2])
    FS_mid.set_variable("obj", user_inputs[3])
    FS_mid.set_variable("wards", user_inputs[4])
    FS_mid.set_variable("gold", user_inputs[5])

    # Perform Mamdani inference and print output
    print(FS_mid.Mamdani_inference(["score"], ignore_errors=False, ignore_warnings=False, verbose=True))

    # Plotting surface
    xs = []
    ys = []
    zs = []
    DIVs = 20
    for x in linspace(0,10,DIVs):
        for y in linspace(0,500,DIVs):
            FS_mid.set_variable("cs", x)
            FS_mid.set_variable("gold", y)
            tip = FS_mid.inference()['score']
            xs.append(x)
            ys.append(y)
            zs.append(tip)
    xs = array(xs)
    ys = array(ys)
    zs = array(zs)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx, yy = plt.meshgrid(xs,ys)

    ax.plot_trisurf(xs,ys,zs, vmin=0, vmax=25, cmap='gnuplot2')
    ax.set_xlabel("CS/min")
    ax.set_ylabel("Gold/min")
    ax.set_zlabel("Score")
    ax.set_title("Mid: CS/min x Gold/min x Score", pad=20)
    ax.set_zlim(0, 1)
    plt.tight_layout()
    plt.show()

elif role_number == 3:

    # FUZZY SYSTEM FOR BOT
    # Create a fuzzy system object
    FS_bot = FuzzySystem()

    FS_bot.add_linguistic_variable("cs", LinguisticVariable([cs_1, cs_2, cs_3], concept=features_names[0], universe_of_discourse=[0,10]))
    FS_bot.add_linguistic_variable("kd", LinguisticVariable([kd_1, kd_2, kd_3], concept=features_names[1], universe_of_discourse=[0,10]))
    FS_bot.add_linguistic_variable("kda", LinguisticVariable([kda_1, kda_2, kda_3], concept=features_names[2], universe_of_discourse=[0,30]))
    FS_bot.add_linguistic_variable("obj", LinguisticVariable([obj_1, obj_2, obj_3], concept=features_names[3], universe_of_discourse=[0,30]))
    FS_bot.add_linguistic_variable("wards", LinguisticVariable([wards_1, wards_2, wards_3], concept=features_names[4], universe_of_discourse=[0,1.5]))
    FS_bot.add_linguistic_variable("gold", LinguisticVariable([gold_1, gold_2, gold_3], concept=features_names[5], universe_of_discourse=[0,500]))
    FS_bot.add_linguistic_variable("score", LinguisticVariable([score_1, score_2, score_3, score_4], universe_of_discourse=[0,1]))

    # Define fuzzy rules for Bot
    fuzzy_rules_bot = ["IF (cs IS Low) AND (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Medium) AND (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Low) AND (gold IS Medium) THEN (score IS Poor)",
                    "IF (cs IS Medium) AND (gold IS Medium) THEN (score IS Good)",
                    "IF (cs IS Medium) AND (gold IS High) THEN (score IS Excellent)",
                    "IF (cs IS High) AND (gold IS Medium) THEN (score IS Excellent)",
                    "IF (cs IS High) AND (gold IS High) THEN (score IS Excellent)",

                    "IF (cs IS Low) OR (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Medium) OR (gold IS Low) THEN (score IS Poor)",
                    "IF (cs IS Low) OR (gold IS Medium) THEN (score IS Poor)",
                    "IF (cs IS Medium) OR (gold IS Medium) THEN (score IS Average)",
                    "IF (cs IS Medium) OR (gold IS High) THEN (score IS Good)",
                    "IF (cs IS High) OR (gold IS Medium) THEN (score IS Good)",
                    "IF (cs IS High) OR (gold IS High) THEN (score IS Excellent)",

                    "IF (kd IS Low) AND (obj IS Low) THEN (score IS Poor)",
                    "IF (kd IS Medium) AND (obj IS Low) THEN (score IS Average)",
                    "IF (kd IS Low) AND (obj IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) AND (obj IS Medium) THEN (score IS Good)",
                    "IF (kd IS Medium) AND (obj IS High) THEN (score IS Good)",
                    "IF (kd IS High) AND (obj IS Medium) THEN (score IS Good)",
                    "IF (kd IS High) AND (obj IS High) THEN (score IS Excellent)",

                    "IF (kd IS Low) OR (obj IS Low) THEN (score IS Poor)",
                    "IF (kd IS Medium) OR (obj IS Low) THEN (score IS Average)",
                    "IF (kd IS Low) OR (obj IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) OR (obj IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) OR (obj IS High) THEN (score IS Good)",
                    "IF (kd IS High) OR (obj IS Medium) THEN (score IS Good)",
                    "IF (kd IS High) OR (obj IS High) THEN (score IS Good)",

                    "IF (kda IS Low) AND (wards IS Low) THEN (score IS Poor)",
                    "IF (kda IS Medium) AND (wards IS Low) THEN (score IS Average)",
                    "IF (kda IS Low) AND (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) AND (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) AND (wards IS High) THEN (score IS Average)",
                    "IF (kda IS High) AND (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS High) AND (wards IS High) THEN (score IS Good)"
                    
                    "IF (kda IS Low) OR (wards IS Low) THEN (score IS Poor)",
                    "IF (kda IS Medium) OR (wards IS Low) THEN (score IS Average)",
                    "IF (kda IS Low) OR (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) OR (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) OR (wards IS High) THEN (score IS Average)",
                    "IF (kda IS High) OR (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS High) OR (wards IS High) THEN (score IS Good)"]

    FS_bot.add_rules(fuzzy_rules_bot)

    # Set antecedents values
    FS_bot.set_variable("cs", user_inputs[0])
    FS_bot.set_variable("kd", user_inputs[1])
    FS_bot.set_variable("kda", user_inputs[2])
    FS_bot.set_variable("obj", user_inputs[3])
    FS_bot.set_variable("wards", user_inputs[4])
    FS_bot.set_variable("gold", user_inputs[5])

    # Perform Mamdani inference and print output
    print(FS_bot.Mamdani_inference(["score"], ignore_errors=False, ignore_warnings=False, verbose=True))

    # Plotting surface
    xs = []
    ys = []
    zs = []
    DIVs = 20
    for x in linspace(0,10,DIVs):
        for y in linspace(0,500,DIVs):
            FS_bot.set_variable("cs", x)
            FS_bot.set_variable("gold", y)
            tip = FS_bot.inference()['score']
            xs.append(x)
            ys.append(y)
            zs.append(tip)
    xs = array(xs)
    ys = array(ys)
    zs = array(zs)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx, yy = plt.meshgrid(xs,ys)

    ax.plot_trisurf(xs,ys,zs, vmin=0, vmax=25, cmap='gnuplot2')
    ax.set_xlabel("CS/min")
    ax.set_ylabel("Gold/min")
    ax.set_zlabel("Score")
    ax.set_title("Bot: CS/min x Gold/min x Score", pad=20)
    ax.set_zlim(0, 1)
    plt.tight_layout()
    plt.show()

elif role_number == 4:

    # FUZZY SYSTEM FOR SUPPORT
    # Create a fuzzy system object
    FS_sup = FuzzySystem()

    FS_sup.add_linguistic_variable("cs", LinguisticVariable([cs_1, cs_2, cs_3], concept=features_names[0], universe_of_discourse=[0,10]))
    FS_sup.add_linguistic_variable("kd", LinguisticVariable([kd_1, kd_2, kd_3], concept=features_names[1], universe_of_discourse=[0,10]))
    FS_sup.add_linguistic_variable("kda", LinguisticVariable([kda_1, kda_2, kda_3], concept=features_names[2], universe_of_discourse=[0,30]))
    FS_sup.add_linguistic_variable("obj", LinguisticVariable([obj_1, obj_2, obj_3], concept=features_names[3], universe_of_discourse=[0,30]))
    FS_sup.add_linguistic_variable("wards", LinguisticVariable([wards_1, wards_2, wards_3], concept=features_names[4], universe_of_discourse=[0,1.5]))
    FS_sup.add_linguistic_variable("gold", LinguisticVariable([gold_1, gold_2, gold_3], concept=features_names[5], universe_of_discourse=[0,500]))
    FS_sup.add_linguistic_variable("score", LinguisticVariable([score_1, score_2, score_3, score_4], universe_of_discourse=[0,1]))

    # Define fuzzy rules for Support
    fuzzy_rules_sup = ["IF (kda IS Low) AND (wards IS Low) THEN (score IS Poor)",
                    "IF (kda IS Medium) AND (wards IS Low) THEN (score IS Poor)",
                    "IF (kda IS Low) AND (wards IS Medium) THEN (score IS Poor)",
                    "IF (kda IS Medium) AND (wards IS Medium) THEN (score IS Good)",
                    "IF (kda IS Medium) AND (wards IS High) THEN (score IS Excellent)",
                    "IF (kda IS High) AND (wards IS Medium) THEN (score IS Excellent)",
                    "IF (kda IS High) AND (wards IS High) THEN (score IS Excellent)",

                    "IF (kda IS Low) OR (wards IS Low) THEN (score IS Poor)",
                    "IF (kda IS Medium) OR (wards IS Low) THEN (score IS Poor)",
                    "IF (kda IS Low) OR (wards IS Medium) THEN (score IS Poor)",
                    "IF (kda IS Medium) OR (wards IS Medium) THEN (score IS Average)",
                    "IF (kda IS Medium) OR (wards IS High) THEN (score IS Good)",
                    "IF (kda IS High) OR (wards IS Medium) THEN (score IS Good)",
                    "IF (kda IS High) OR (wards IS High) THEN (score IS Excellent)",

                    "IF (kd IS Low) AND (obj IS Low) THEN (score IS Poor)",
                    "IF (kd IS Medium) AND (obj IS Low) THEN (score IS Average)",
                    "IF (kd IS Low) AND (obj IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) AND (obj IS Medium) THEN (score IS Good)",
                    "IF (kd IS Medium) AND (obj IS High) THEN (score IS Good)",
                    "IF (kd IS High) AND (obj IS Medium) THEN (score IS Good)",
                    "IF (kd IS High) AND (obj IS High) THEN (score IS Excellent)",

                    "IF (kd IS Low) OR (obj IS Low) THEN (score IS Poor)",
                    "IF (kd IS Medium) OR (obj IS Low) THEN (score IS Average)",
                    "IF (kd IS Low) OR (obj IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) OR (obj IS Medium) THEN (score IS Average)",
                    "IF (kd IS Medium) OR (obj IS High) THEN (score IS Good)",
                    "IF (kd IS High) OR (obj IS Medium) THEN (score IS Good)",
                    "IF (kd IS High) OR (obj IS High) THEN (score IS Good)",

                    "IF (cs IS Low) AND (kd IS Low) THEN (score IS Poor)",
                    "IF (cs IS Medium) AND (kd IS Low) THEN (score IS Average)",
                    "IF (cs IS Low) AND (kd IS Medium) THEN (score IS Average)",
                    "IF (cs IS Medium) AND (kd IS Medium) THEN (score IS Average)",
                    "IF (cs IS Medium) AND (kd IS High) THEN (score IS Average)",
                    "IF (cs IS High) AND (kd IS Medium) THEN (score IS Average)",
                    "IF (cs IS High) AND (kd IS High) THEN (score IS Good)"
                    
                    "IF (cs IS Low) OR (kd IS Low) THEN (score IS Poor)",
                    "IF (cs IS Medium) OR (kd IS Low) THEN (score IS Average)",
                    "IF (cs IS Low) OR (kd IS Medium) THEN (score IS Average)",
                    "IF (cs IS Medium) OR (kd IS Medium) THEN (score IS Average)",
                    "IF (cs IS Medium) OR (kd IS High) THEN (score IS Average)",
                    "IF (cs IS High) OR (kd IS Medium) THEN (score IS Average)",
                    "IF (cs IS High) OR (kd IS High) THEN (score IS Good)"]

    FS_sup.add_rules(fuzzy_rules_sup)

    # Set antecedents values
    FS_sup.set_variable("cs", user_inputs[0])
    FS_sup.set_variable("kd", user_inputs[1])
    FS_sup.set_variable("kda", user_inputs[2])
    FS_sup.set_variable("obj", user_inputs[3])
    FS_sup.set_variable("wards", user_inputs[4])
    FS_sup.set_variable("gold", user_inputs[5])

    # Perform Mamdani inference and print output
    print(FS_sup.Mamdani_inference(["score"], ignore_errors=False, ignore_warnings=False, verbose=True))

    # Plotting surface
    xs = []
    ys = []
    zs = []
    DIVs = 20
    for x in linspace(0,30,DIVs):
        for y in linspace(0,1.5,DIVs):
            FS_sup.set_variable("kda", x)
            FS_sup.set_variable("wards", y)
            tip = FS_sup.inference()['score']
            xs.append(x)
            ys.append(y)
            zs.append(tip)
    xs = array(xs)
    ys = array(ys)
    zs = array(zs)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx, yy = plt.meshgrid(xs,ys)

    ax.plot_trisurf(xs,ys,zs, vmin=0, vmax=25, cmap='gnuplot2')
    ax.set_xlabel("K/D/A Ratio")
    ax.set_ylabel("Wards/min")
    ax.set_zlabel("Score")
    ax.set_title("Support: K/D/A Ratio x Wards/min x Score", pad=20)
    ax.set_zlim(0, 1)
    plt.tight_layout()
    plt.show()
