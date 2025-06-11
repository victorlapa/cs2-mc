import pandas as pd

# Load the CSV
df = pd.read_csv("cs2_matches.csv", sep=";")

# Team to check
team_name = "Astralis"

# Count appearances as team1 or team2
count = ((df['team1'] == team_name) | (df['team2'] == team_name)).sum()

print(f"{team_name} appears in {count} matches.")
