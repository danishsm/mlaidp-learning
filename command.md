initialize git repository
    git init

Update your name and email as you to see in commit  
    git config user.name "danish"
    git config user.email "examples@gmail.com"

Check if author and email is set
    git config --list

    git config user.name
    git config user.email

Add files for tracking. Execute below and then status of files will change to 'A' which mean they are added for tracking.
    git add .

Commit files 
    git commit -m "first commit"

Add a branch named main
    git branch -M main

Add remote github repository
    git remote add origin https://github.com/danishsm/mlaidp-learning.git

Push to github . Changes will now reflect in github.
    git push -u origin main


Create a python virtual environment
    python -m venv .venv


Execute below to activate the virtual environment
    .\.venv\Scripts\activate

Install from requirement.txt
    pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org -r requirements.txt