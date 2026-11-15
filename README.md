# ML_Assignment

Below is a complete `README.md` tutorial you can put in your GitHub repository. It documents the problems you encountered, explains why they happened, and provides step-by-step solutions using Git LFS.

\# How to Fix GitHub Push Errors Caused by Large Files (Git LFS)\
\
\## 1. Introduction\
\
When uploading a machine learning project to GitHub, you may encounter errors because some datasets are too large for GitHub's normal Git storage.\
\
This tutorial explains how to solve common Git push problems, including:\
\
1\. GitHub rejecting files larger than 100 MB.\
2\. GitHub recommending Git Large File Storage (Git LFS).\
3\. Git failing to connect to GitHub through HTTPS.\
4\. Git LFS tracking files while Git history still contains large normal Git objects.\
5\. Local and remote branches diverging after rewriting Git history.\
\
The examples in this tutorial use a machine learning project containing large CSV datasets.\
\
\---\
\
\## 2. Understanding the GitHub File Size Limit\
\
GitHub enforces the following file size limits:\
\
\| File size | GitHub behaviour |\
\|---|---|\
\| Below 50 MB | Accepted normally |\
\| 50–100 MB | Warning displayed |\
\| Above 100 MB | Push rejected |\
\
For example, the following error may appear:\
\
\`\`\`text\
remote: warning: File Data_Set/logistics_predictive_maintenanceV2.csv\
is 93.56 MB; this is larger than GitHub's recommended maximum\
file size of 50.00 MB\
\
remote: error: File Data_Set/processed_dataset.csv\
is 201.46 MB; this exceeds GitHub's file size limit of 100.00 MB\
\
remote: error: GH001: Large files detected.\
remote: pre-receive hook declined\
\`\`\`\
\
\*\*Why does this happen?\*\*\
\
GitHub does not allow normal Git repositories to receive individual files larger than 100 MB.\
\
Machine learning datasets, processed CSV files, images and model files can easily exceed this limit.\
\
The recommended solution is Git Large File Storage (Git LFS).\
\
\---\
\
\## 3. Step 1 – Install Git LFS\
\
Git LFS is an extension that allows Git to manage large files separately from normal Git objects.\
\
First, install Git LFS:\
\
https\://git-lfs.com/\
\
After installation, open PowerShell or the PyCharm terminal.\
\
Run:\
\
\`\`\`powershell\
git lfs install\
\`\`\`\
\
This initializes Git LFS on your computer.\
\
\---\
\
\## 4. Step 2 – Configure Git LFS Tracking\
\
Navigate to your project directory:\
\
\`\`\`powershell\
cd "C:\Users\tanha\PycharmProjects\ML_Assignment(GitHub)"\
\`\`\`\
\
Configure Git LFS to track CSV files:\
\
\`\`\`powershell\
git lfs track "\*.csv"\
\`\`\`\
\
You can also track CSV files specifically inside the dataset folder:\
\
\`\`\`powershell\
git lfs track "Data_Set/\*.csv"\
\`\`\`\
\
These commands create or update a \`.gitattributes\` file.\
\
The file should contain tracking rules similar to:\
\
\`\`\`text\
\*.csv filter=lfs diff=lfs merge=lfs -text\
Data_Set/\*.csv filter=lfs diff=lfs merge=lfs -text\
\`\`\`\
\
Add the configuration file to Git:\
\
\`\`\`powershell\
git add .gitattributes\
\`\`\`\
\
\*\*Important:\*\* Git LFS tracking rules only affect files added or re-added after the rules are configured. They do not automatically convert previously committed large files.\
\
\---\
\
\## 5. Step 3 – Check Git LFS Tracking\
\
Run:\
\
\`\`\`powershell\
git lfs track\
\`\`\`\
\
This displays the configured tracking patterns.\
\
Next, run:\
\
\`\`\`powershell\
git lfs ls-files\
\`\`\`\
\
This displays files currently stored using Git LFS.\
\
Example:\
\
\`\`\`text\
4b24744df0 \* Data_Set/logistics_predictive_maintenanceV2(filtered).csv\
279cafcaa5 \* Data_Set/logistics_predictive_maintenanceV2.csv\
ac4fd8cddb \* Data_Set/processed_dataset.csv\
\`\`\`\
\
You can also check whether a specific file has the LFS filter:\
\
\`\`\`powershell\
git check-attr filter -- Data_Set/processed_dataset.csv\
\`\`\`\
\
Expected result:\
\
\`\`\`text\
Data_Set/processed_dataset.csv: filter: lfs\
\`\`\`\
\
If the result is \`filter: lfs\`, the file matches the Git LFS tracking rules.\
\
However, this alone does not prove that every historical commit has been converted to LFS.\
\
\---\
\
\## 6. Step 4 – Understand Why GitHub May Still Reject the File\
\
Even after configuring Git LFS, GitHub may continue rejecting a large file.\
\
For example:\
\
\`\`\`text\
remote: error: File Data_Set/processed_dataset.csv\
is 126.22 MB; this exceeds GitHub's file size limit of 100.00 MB\
\`\`\`\
\
This can happen because the file was committed to normal Git storage before Git LFS tracking was configured.\
\
Git stores project history, meaning previous versions of files can remain inside earlier commits.\
\
Therefore, simply adding a file to Git LFS does not remove its previous normal Git versions from the commit history.\
\
\*\*Solution: Migrate the existing Git history.\*\*\
\
\---\
\
\## 7. Step 5 – Migrate Existing Large Files to Git LFS\
\
Before rewriting Git history, make sure your work is committed and your working directory is clean.\
\
Check:\
\
\`\`\`powershell\
git status\
\`\`\`\
\
If necessary, create a backup of your repository first.\
\
Then migrate the problematic dataset:\
\
\`\`\`powershell\
git lfs migrate import --include="Data_Set/processed_dataset.csv" --everything\
\`\`\`\
\
Explanation:\
\
\- \`git lfs migrate import\`: Converts existing Git file versions into Git LFS objects.\
\- \`--include\`: Specifies which file should be migrated.\
\- \`--everything\`: Applies the migration across the Git history and relevant references.\
\
Example output:\
\
\`\`\`text\
Sorting commits: ..., done.\
Rewriting commits: 100% (8/8), done.\
Updating refs: ..., done.\
Checkout: ..., done.\
\`\`\`\
\
This means the migration completed successfully.\
\
\*\*Important:\*\* Migrating Git history changes commit IDs. This is why your local and remote Git branches may diverge afterward.\
\
If you need to migrate multiple files, you can use an appropriate include pattern, such as:\
\
\`\`\`powershell\
git lfs migrate import --include="Data_Set/\*.csv" --everything\
\`\`\`\
\
Only migrate the files you intend to store using Git LFS.\
\
\---\
\
\## 8. Step 6 – Resolve Diverged Git Branches\
\
After migrating Git history, you may encounter:\
\
\`\`\`text\
On branch main\
\
Your branch and 'origin/main' have diverged,\
and have 8 and 2 different commits each, respectively.\
\
nothing to commit, working tree clean\
\`\`\`\
\
\*\*Why does this happen?\*\*\
\
Git LFS migration rewrites local commits to replace normal Git file objects with LFS pointers.\
\
The remote GitHub repository still contains the previous commit history.\
\
Consequently, Git sees the local and remote branches as different histories.\
\
Check the current branch:\
\
\`\`\`powershell\
git branch\
\`\`\`\
\
Check the remote configuration:\
\
\`\`\`powershell\
git remote -v\
\`\`\`\
\
If the migration is correct and the remote-only commits do not contain work that needs preserving, update the remote branch using:\
\
\`\`\`powershell\
git push --force-with-lease origin main\
\`\`\`\
\
\### Why use --force-with-lease?\
\
A normal push may fail because the rewritten local history is no longer a direct continuation of the remote history.\
\
\`--force-with-lease\` allows Git to replace the remote branch history while checking that the remote reference has not changed unexpectedly since your last fetch.\
\
\*\*Warning:\*\* This operation rewrites the remote branch history. Coordinate with collaborators and preserve any remote-only commits before proceeding. Do not use it blindly on shared repositories.\
\
Avoid running \`git pull\` just to resolve this divergence, because merging the old history back may reintroduce the oversized Git objects.\
\
\---\
\
\## 9. Step 7 – Troubleshoot GitHub Connection Errors\
\
Another possible error is:\
\
\`\`\`text\
fatal: unable to access\
'https\://github.com/...':\
\
Failed to connect to github.com port 443\
Could not connect to server\
\`\`\`\
\
This is a network connectivity issue rather than a file size issue.\
\
Check your connection:\
\
\`\`\`powershell\
Test-NetConnection github.com -Port 443\
\`\`\`\
\
Look for:\
\
\`\`\`text\
TcpTestSucceeded : True\
\`\`\`\
\
If it returns \`False\`, check:\
\
\- Internet connectivity.\
\- Firewall settings.\
\- VPN or proxy configuration.\
\- Temporary GitHub connectivity problems.\
\
You can also open https\://github.com in your browser to check whether the website is accessible.\
\
If an incorrect proxy is configured, inspect it:\
\
\`\`\`powershell\
git config --global --get http.proxy\
git config --global --get https.proxy\
\`\`\`\
\
Only remove proxy settings if they are incorrect or no longer needed.\
\
\---\
\
\## 10. Step 8 – Verify the Final Upload\
\
After successfully pushing, verify that the repository is synchronized.\
\
Run:\
\
\`\`\`powershell\
git lfs ls-files\
\`\`\`\
\
Check that the intended large CSV files appear.\
\
Then:\
\
\`\`\`powershell\
git status\
\`\`\`\
\
Expected output:\
\
\`\`\`text\
On branch main\
Your branch is up to date with 'origin/main'.\
\
nothing to commit, working tree clean\
\`\`\`\
\
Finally, open your GitHub repository and check that your files are available.\
\
Repository:\
\
https\://github.com/TANHAN-YUAN/ML_Assignment\
\
\---\
\
\## 11. Common Git Commands Summary\
\
\| Command | Purpose |\
\|---|---|\
\| \`git lfs install\` | Initialize Git LFS |\
\| \`git lfs track "\*.csv"\` | Track CSV files using LFS |\
\| \`git add .gitattributes\` | Add LFS tracking configuration |\
\| \`git lfs ls-files\` | List LFS-managed files |\
\| \`git check-attr filter -- filename\` | Check whether a file uses LFS |\
\| \`git status\` | Check working directory and branch status |\
\| \`git log --oneline --all\` | Inspect commit history |\
\| \`git lfs migrate import --include="filename" --everything\` | Migrate existing file history to LFS |\
\| \`git push origin main\` | Push local commits |\
\| \`git push --force-with-lease origin main\` | Update remote history after a migration, when safe |\
\
\---\
\
\## 12. Conclusion\
\
Uploading large datasets is a common challenge when working on machine learning projects with GitHub.\
\
The main lesson is that configuring Git LFS is not always enough. If large files were committed before Git LFS was configured, their previous versions may remain in Git history.\
\
The correct approach is to:\
\
1\. Install and initialize Git LFS.\
2\. Configure the appropriate file tracking rules.\
3\. Verify that the datasets are tracked by Git LFS.\
4\. Migrate existing large files from normal Git history when necessary.\
5\. Resolve the resulting local and remote history divergence carefully.\
6\. Push the corrected history to GitHub.\
7\. Verify that the repository is synchronized.\
\
Using Git LFS helps keep machine learning projects manageable while allowing large datasets to be stored separately from normal Git objects.\
\
\*\*Remember:\*\* Always back up your repository before rewriting Git history, and check your GitHub LFS storage and bandwidth quotas before uploading large datasets.\

**My suggestion:** Save this as `README.md` in your project root. It will serve as both a troubleshooting guide and technical documentation explaining how you resolved the GitHub upload issue.

One important point: the README documents the complete solution, but your final `git push --force-with-lease origin main` still needs to succeed before you can consider the repository upload finished.****
