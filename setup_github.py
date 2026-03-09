import os
import requests
import subprocess
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
REPO_NAME = "pintrade"

def create_github_repo():
    if not GITHUB_TOKEN or not GITHUB_USERNAME:
        print("Error: GITHUB_TOKEN or GITHUB_USERNAME not set in .env")
        return False

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "name": REPO_NAME,
        "description": "PINtrade: Alpha factor research and backtesting engine",
        "private": False,
        "auto_init": False # We will initialize it manually
    }
    response = requests.post("https://api.github.com/user/repos", headers=headers, json=data)

    if response.status_code == 201:
        print(f"Repository '{REPO_NAME}' created successfully on GitHub.")
        return True
    elif response.status_code == 422 and "name already exists" in response.text:
        print(f"Repository '{REPO_NAME}' already exists on GitHub. Skipping creation.")
        return True
    else:
        print(f"Error creating repository: {response.status_code} - {response.text}")
        return False

def setup_git():
    if not GITHUB_USERNAME:
        print("Error: GITHUB_USERNAME not set in .env")
        return

    repo_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

    commands = [
        "git init",
        "git add .",
        "git commit -m \"Initial commit for PINtrade project\"",
        "git branch -M main",
        "git push -u origin main"
    ]

    # Check if remote 'origin' already exists
    try:
        subprocess.run("git remote get-url origin", shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        print("Remote 'origin' already exists. Setting its URL.")
        subprocess.run(f"git remote set-url origin {repo_url}", shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
    except subprocess.CalledProcessError:
        print("Remote 'origin' does not exist. Adding it.")
        commands.insert(3, f"git remote add origin {repo_url}") # Insert after commit

    for command in commands:
        try:
            print(f"Running: {command}")
            subprocess.run(command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        except subprocess.CalledProcessError as e:
            print(f"Error running git command: {command}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            if e.stdout and "Everything up-to-date" in e.stdout and "git push" in command:
                print("Repository already up-to-date.")
            elif e.stderr and "Your branch is up to date with 'origin/main'" in e.stderr and "git push" in command:
                print("Repository already up-to-date.")
            else:
                print(f"Failed to execute command: {command}")
                return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return
    print("Git repository initialized and pushed to GitHub.")

if __name__ == "__main__":
    print("Starting GitHub setup...")
    if create_github_repo():
        setup_git()
    print("GitHub setup complete.")