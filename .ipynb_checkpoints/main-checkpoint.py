import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

classData = pd.read_csv("data\classData.csv")
classData.head()

def fetch_github_data(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch data from GitHub. Ensure username is correct and connection is available.")
        return None
    
    repos_data = response.json()
    projects = []
    for repo in repos_data:
        project_info = {
            'name': repo['name'],
            'description': repo['description'] or '',
            'language': repo['language']
        }
        projects.append(project_info)
    return pd.DataFrame(projects)

def get_user_preferences():
    interests_options = [
        "Machine Learning",
    "Web Development",
    "Data Science",
    "UI/UX Design",
    "Cybersecurity",
    "Cloud Computing",
    "Game Development",
    "Data Engineering",
    "Robotics",
    "Blockchain"
    ]
    print("Choose your topics of interest (enter number, separated by commas if more than one):")
    for i, option in enumerate(interests_options, start=1):
        print(f"{i}. {option}")
    
    interests_input = input("Your choices: ")
    interests_indices = [int(index.strip()) - 1 for index in interests_input.split(',') if index.strip().isdigit()]
    interests = [interests_options[i] for i in interests_indices if 0 <= i < len(interests_options)]

    skills_options = [
            "Python", 
    "JavaScript", 
    "PHP",
    "Java", 
    "R", 
    "SQL", 
    "C#", 
    "Ruby", 
    "Dart", 
    "Solidity", 
    "Docker", 
    "YAML", 
    "Scala", 
    "C++", 
    "HTML", 
    "CSS",
    "Tensorflow"
        ]
    print("\nChoose skills you want to develop (enter number, separated by commas if more than one):")
    for i, option in enumerate(skills_options, start=1):
        print(f"{i}. {option}")
    
    skills_input = input("Your choices: ")
    skills_indices = [int(index.strip()) - 1 for index in skills_input.split(',') if index.strip().isdigit()]
    desired_skills = [skills_options[i] for i in skills_indices if 0 <= i < len(skills_options)]

    preferences = {
        'interests': interests,
        'desired_skills': desired_skills
    }
    return preferences

